import datetime
import time
from pathlib import Path

import colossalai
import timm
import timm.optim.optim_factory as optim_factory
import torch
import torchvision.datasets as datasets
from colossalai.context import Config
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.nn.lr_scheduler import CosineAnnealingLR
from colossalai.utils import get_dataloader
from colossalai.utils.checkpointing import load_checkpoint, save_checkpoint
from torchvision import transforms
from tqdm import tqdm

import models_mae_tp

assert timm.__version__ == "0.3.2"  # version check


# global states
LOGGER = get_dist_logger()
VERBOSE = False
DEBUG = False


def _load_imgfolder(path, transform):
    return datasets.ImageFolder(path, transform=transform)


def model(norm_pix_loss):
    m = models_mae_tp.mae_vit_large_patch16(norm_pix_loss=norm_pix_loss)
    if VERBOSE:
        LOGGER.info("Use model vit_large_patch16")
    return m


def criterion():
    c = torch.nn.CrossEntropyLoss()
    if VERBOSE:
        LOGGER.info(f"Criterion:\n{c}")
    return c


def optimizer(model, learning_rate, weight_decay):
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model, weight_decay)
    o = torch.optim.AdamW(param_groups, lr=learning_rate, betas=(0.9, 0.95))
    if VERBOSE:
        LOGGER.info(f"Optimizer:\n{o}")
    return o


def pretrain_dataloaders(
    datapath: Path,
    transform_train: transforms.Compose,
    transform_val: transforms.Compose,
):
    if VERBOSE:
        LOGGER.info(f"DATAPATH: {datapath.absolute()}")
    train_dataset = _load_imgfolder(datapath / "train", transform_train)
    test_dataset = _load_imgfolder(datapath / "val", transform_val)
    if VERBOSE:
        LOGGER.info(f"Train dataset:\n{train_dataset}")
        LOGGER.info(f"Test dataset:\n{test_dataset}")

    train_dataloader = get_dataloader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=gpc.config.BATCH_SIZE,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
    )

    test_dataloader = get_dataloader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=gpc.config.BATCH_SIZE,
        num_workers=1,
        pin_memory=True,
        drop_last=False,
    )

    return train_dataloader, test_dataloader


def init_global_states(config: Config):
    global VERBOSE, DEBUG
    VERBOSE = config.VERBOSE
    DEBUG = config.DEBUG


def init_engine(config: Config):
    _model = model(config.NORM_PIX_LOSS)
    _optimizer = optimizer(_model, config.LEARNING_RATE, config.WEIGHT_DECAY)
    _criterion = criterion()
    train_dataloader, test_dataloader = pretrain_dataloaders(
        config.DATAPATH, config.TRANSFORM_TRAIN, config.TRANSFORM_VAL
    )
    engine, train_dataloader, test_dataloader, _ = colossalai.initialize(
        _model,
        _optimizer,
        _criterion,
        train_dataloader,
        test_dataloader,
    )
    return engine, train_dataloader, test_dataloader


def scale_loss(engine, loss, loss_scaler, data_iter_step, config):
    loss /= config.ACCUM_ITER
    loss_scaler(
        loss,
        engine.optimizer,
        parameters=engine.model.parameters(),
        update_grad=(data_iter_step + 1) % config.ACCUM_ITER == 0,
    )


def save_model(model, output_dir, epoch):
    checkpoint_path = output_dir / (f"checkpoint-{epoch}.pth")
    save_checkpoint(checkpoint_path, epoch, model, None, None)


def main(config_path):
    colossalai.launch_from_torch(config_path)
    config = gpc.config

    init_global_states(config)
    engine, train_dataloader, _ = init_engine(config)
    lr_scheduler = CosineAnnealingLR(engine.optimizer, total_steps=config.NUM_EPOCHS)

    start_epoch = 0
    if config.RESUME:
        # WARNING: `load_checkpoint()` and `save_checkpoint()`
        #          won't touch optimizer and lr_scheduler!
        start_epoch = 1 + load_checkpoint(
            config.RESUME_DIR, engine.model, _, _, strict=False
        )
        LOGGER.info(
            f"Resume from checkpoint {config.RESUME_DIR}, start epoch {start_epoch}"
        )

    LOGGER.info(f"Start pre-training for {config.NUM_EPOCHS} epochs")
    start_time = time.time()
    for epoch in range(start_epoch, config.NUM_EPOCHS):

        engine.train()
        # TODO: This part could be more "colossal-native", like construct a correct `engine.criterion`.
        for idx, (img, _) in enumerate(tqdm(train_dataloader, desc=f"epoch {epoch}")):
            # we use a per iteration (instead of per epoch) lr scheduler
            img = img.cuda()

            engine.zero_grad()
            loss, _, _ = engine.model(img, mask_ratio=config.MASK_RATIO)
            engine.backward(loss)
            engine.step()
            lr_scheduler.step()

            if DEBUG:
                LOGGER.info(
                    f"iteration {idx}, first 10 elements of param: {next(engine.model.parameters()).flatten()[:10]}"
                )

        if config.OUTPUT_DIR and (
            epoch % config.CHECKPOINT_INTERVAL == 0 or epoch + 1 == config.NUM_EPOCHS
        ):
            save_model(
                engine.model,
                config.OUTPUT_DIR,
                epoch,
            )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    LOGGER.info(f"Training time {total_time_str}")


if __name__ == "__main__":
    args = colossalai.get_default_parser().parse_args()
    if args.config:
        config = args.config
    else:
        config = Path(__file__).parent / "config" / "pretrain.py"

    main(config)
