from pathlib import Path
from timm.utils import accuracy
from tqdm import tqdm
from colossalai.logging import get_dist_logger
import colossalai
import torch
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


def init_pretrain_dataloaders(datapath):
    ...


def main(config_path):
    colossalai.launch_from_torch(config_path)

    logger = get_dist_logger()

    # build mae model
    model = models_vit.vit_large_patch16(
        num_classes=1000,
        global_pool=False,
    )

    # build dataloaders
    datapath = gpc.config.DATAPATH
    train_dataset = load_imgfolder(datapath / "train", gpc.config.TRANSFORM_TRAIN)
    test_dataset = load_imgfolder(datapath / "val", gpc.config.TRANSFORM_VAL)

    print(train_dataset)
    print(test_dataset)

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

    train_dataloader, test_dataloader = init_pretrain_dataloaders(
        gpc.config.DATAPATH, gpc.config.TRANSFORM_TRAIN, gpc.config.TRANSFORM_VAL
    )

    criterion = torch.nn.CrossEntropyLoss()
    print("criterion = {}".format(str(criterion)))

    optimizer = LARS(model.head.parameters(), lr=False, weight_decay=0)
    print(optimizer)

    engine, train_dataloader, test_dataloader, _ = colossalai.initialize(
        model,
        optimizer,
        criterion,
        train_dataloader,
        test_dataloader,
    )

    lr_scheduler = CosineAnnealingLR(optimizer, total_steps=gpc.config.NUM_EPOCHS)

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
        # TODO: replace this with custom loss scaler
        lr_scheduler.step()

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
