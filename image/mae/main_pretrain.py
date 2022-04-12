from os import PathLike
from pathlib import Path

import colossalai
import timm
import timm.optim.optim_factory as optim_factory
import torch
import torchvision.datasets as datasets
from colossalai.context import Config
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.utils import get_dataloader
from timm.utils import accuracy
from torchvision import transforms
from tqdm import tqdm

import models_vit
import util.lr_sched as lr_sched
import util.misc as misc
from deit_helper import load_model_args, lr_sched_args
from util.misc import NativeScalerWithGradNormCount as NativeScaler

assert timm.__version__ == "0.3.2"  # version check


# global states
LOGGER = get_dist_logger()
VERBOSE = False


def _load_imgfolder(path, transform):
    return datasets.ImageFolder(path, transform=transform)


def model():
    m = models_vit.vit_large_patch16(
        num_classes=1000,
        global_pool=False,
    )
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
    global VERBOSE
    VERBOSE = config.VERBOSE


def init_engine(config: Config):
    _model = model()
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


def resume_model(engine, loss_scaler, resume_address, start_epoch):
    misc.load_model(
        args=load_model_args(resume=resume_address, start_epoch=start_epoch),
        model_without_ddp=engine.model,
        optimizer=engine.optimizer,
        loss_scaler=loss_scaler,
    )
    if VERBOSE:
        LOGGER.info(f"Resume model from {resume_address}, start at epoch {start_epoch}")


def main(config_path):
    colossalai.launch_from_torch(config_path)
    config = gpc.config

    init_global_states(config)
    engine, train_dataloader, test_dataloader = init_engine(config)
    loss_scaler = NativeScaler()

    if config.RESUME:
        resume_model(
            engine, loss_scaler, config.RESUME_ADDRESS, config.RESUME_START_EPOCH
        )

    LOGGER.info(f"Start pre-training for {config.NUM_EPOCHS} epochs")
    for epoch in range(config.NUM_EPOCHS):
        engine.train()
        engine.zero_grad()

        # display progress bar if main
        if gpc.get_global_rank() == 0:
            train_dl = tqdm(train_dataloader)
        else:
            train_dl = train_dataloader

        for data_iter_step, (samples, target) in enumerate(train_dl):
            # TODO: handle learning rate adjustion more properly.
            # # we use a per iteration (instead of per epoch) lr scheduler
            # if data_iter_step % ACCUM_ITER == 0:
            #     lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(train_dl) + epoch, LR_SCHED_ARGS)

            samples = samples.cuda()
            target = target.cuda()
            output = engine(samples)
            train_loss = engine.criterion(output, target)
            engine.backward(train_loss)
            engine.step()

            if (data_iter_step + 1) % config.ACCUM_ITER == 0:
                engine.zero_grad()
        # TODO: custom loss scaler

        engine.eval()

        for image, target in test_dataloader:
            image = image.cuda()
            target = target.cuda()

            with torch.no_grad():
                output = engine(image)
                test_loss = engine.criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

        # TODO: smooth average


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        config = Path(__file__).parent / "config" / "pretrain.py"
    else:
        config = sys.argv[1]
    main(config)
