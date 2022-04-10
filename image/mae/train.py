from pathlib import Path
from colossalai.nn.metric import Accuracy
from colossalai.logging import get_dist_logger
import colossalai
import torch
import os
from colossalai.core import global_context as gpc
from colossalai.utils import get_dataloader, MultiTimer
from colossalai.trainer import Trainer, hooks
import torchvision.datasets as datasets
import util.lr_sched as lr_sched
from torchvision import transforms
from colossalai.nn.lr_scheduler import CosineAnnealingLR
from util.lars import LARS
import models_vit
from util.crop import RandomResizedCrop

TRANSFORM_TRAIN = transforms.Compose([
        RandomResizedCrop(224, interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

TRANSFORM_VAL = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def load_imgfolder(path, transform):
    return datasets.ImageFolder(path, transform=transform)

def main():
    colossalai.launch_from_torch(config='./config.py')

    logger = get_dist_logger()

    # build mae model
    model = models_vit.vit_large_patch16(
        num_classes=1000,
        global_pool=False,
    )

    # build dataloaders
    datapath = Path(os.environ['DATA'])
    train_dataset = load_imgfolder(datapath/'train', TRANSFORM_TRAIN)
    test_dataset = load_imgfolder(datapath/'val', TRANSFORM_VAL)

    print(train_dataset)
    print(test_dataset)

    train_dataloader = get_dataloader(
        dataset = train_dataset,
        shuffle=True,
        batch_size=gpc.config.BATCH_SIZE,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
    )

    test_dataloader = get_dataloader(
        dataset = train_dataset,
        shuffle=True,
        batch_size=gpc.config.BATCH_SIZE,
        num_workers=1,
        pin_memory=True,
        drop_last=False
    )

    criterion = torch.nn.CrossEntropyLoss()
    print("criterion = {}".format(str(criterion)))

    optimizer = LARS(model.head.parameters(), lr=False, weight_decay=0)
    print(optimizer)

    engine, train_dataloader, test_dataloader, _ = colossalai.initialize(model,
                                                                         optimizer,
                                                                         criterion,
                                                                         train_dataloader,
                                                                         test_dataloader,
                                                                         )
    # build a timer to measure time
    timer = MultiTimer()

    # create a trainer object
    trainer = Trainer(
        engine=engine,
        timer=timer,
        logger=logger
    )

    lr_scheduler = CosineAnnealingLR(optimizer, total_steps=gpc.config.NUM_EPOCHS)

    # define the hooks to attach to the trainer
    hook_list = [
        hooks.LossHook(),
        hooks.LRSchedulerHook(lr_scheduler=lr_scheduler, by_epoch=False),
        hooks.AccuracyHook(accuracy_func=Accuracy()),
        hooks.LogMetricByEpochHook(logger),
        hooks.LogMemoryByEpochHook(logger),
        hooks.LogTimingByEpochHook(timer, logger),

        # you can uncomment these lines if you wish to use them
        # hooks.TensorboardHook(log_dir='./tb_logs', ranks=[0]),
        # hooks.SaveCheckpointHook(checkpoint_dir='./ckpt')
    ]

    # start training
    trainer.fit(
        train_dataloader=train_dataloader,
        epochs=gpc.config.NUM_EPOCHS,
        test_dataloader=test_dataloader,
        test_interval=1,
        hooks=hook_list,
        display_progress=True
    )


if __name__ == '__main__':
    main()
