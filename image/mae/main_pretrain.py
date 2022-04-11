from pathlib import Path
from timm.utils import accuracy
from tqdm import tqdm
from colossalai.logging import get_dist_logger
import colossalai
import torch
from colossalai.context import Config
from colossalai.core import global_context as gpc
from colossalai.utils import get_dataloader
import torchvision.datasets as datasets
import util.lr_sched as lr_sched
from torchvision import transforms
from colossalai.nn.lr_scheduler import CosineAnnealingLR
from util.lars import LARS
import models_vit
from dataclasses import dataclass

ACCUM_ITER = 1

# global states
LOGGER = get_dist_logger()
VERBOSE = False


@dataclass
class lr_sched_args:
    warmup_epochs: int
    lr: float
    min_lr: float


# FIXME: `lr` should be `absolute_lr = base_lr * total_batch_size / 256`
LR_SCHED_ARGS = lr_sched_args(
    warmup_epochs=1,
    lr=0.1,
    min_lr=0,
)


def load_imgfolder(path, transform):
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


def optimizer(model):
    o = LARS(model.head.parameters(), lr=False, weight_decay=0)
    if VERBOSE:
        LOGGER.info(f"Optimizer:\n{o}")
    return o


def pretrain_dataloaders(datapath, transform_train, transform_val):
    train_dataset = load_imgfolder(datapath / "train", transform_train)
    test_dataset = load_imgfolder(datapath / "val", transform_val)

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
    _optimizer = optimizer(_model)
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


def main(config_path):
    colossalai.launch_from_torch(config_path)

    init_global_states(gpc.config)
    engine, train_dataloader, test_dataloader = init_engine(gpc.config)

    for epoch in range(gpc.config.NUM_EPOCHS):
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

            if (data_iter_step + 1) % ACCUM_ITER == 0:
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
