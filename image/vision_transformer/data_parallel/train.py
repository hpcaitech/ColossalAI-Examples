import glob
import os

import colossalai
import torch
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.lr_scheduler import LinearWarmupLR
from colossalai.trainer import Trainer, hooks
from timm.models import vit_base_patch16_224

from titans.dataloader.imagenet import build_dali_imagenet
from mixup import MixupAccuracy, MixupLoss
from myhooks import TotalBatchsizeHook


def main():
    # initialize distributed setting
    parser = colossalai.get_default_parser()
    args = parser.parse_args()
    disable_existing_loggers()
    # launch from slurm batch job
    # colossalai.launch_from_slurm(config=args.config,
    #                              host=args.host,
    #                              port=args.port,
    #                              backend=args.backend
    #                              )
    # launch from torch
    colossalai.launch_from_torch(config=args.config)

    # get logger
    logger = get_dist_logger()
    logger.info("initialized distributed environment", ranks=[0])

    # build model
    model = vit_base_patch16_224(drop_rate=0.1)

    # build dataloader
    root = os.environ['DATA']
    train_dataloader, test_dataloader = build_dali_imagenet(root, rand_augment=True)
    # build optimizer
    optimizer = colossalai.nn.Lamb(model.parameters(), lr=1.8e-2, weight_decay=0.1)

    # build loss
    criterion = MixupLoss(loss_fn_cls=torch.nn.CrossEntropyLoss)

    # lr_scheduler
    lr_scheduler = LinearWarmupLR(optimizer, warmup_steps=1, total_steps=gpc.config.NUM_EPOCHS)

    engine, train_dataloader, test_dataloader, _ = colossalai.initialize(model, optimizer, criterion, train_dataloader,
                                                                         test_dataloader)
    logger.info("initialized colossalai components", ranks=[0])

    # build trainer
    trainer = Trainer(engine=engine, logger=logger)

    # build hooks
    hook_list = [
        hooks.LossHook(),
        hooks.AccuracyHook(accuracy_func=MixupAccuracy()),
        hooks.LogMetricByEpochHook(logger),
        hooks.LRSchedulerHook(lr_scheduler, by_epoch=True),
        TotalBatchsizeHook(),
    ]

    # start training
    trainer.fit(train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                epochs=gpc.config.NUM_EPOCHS,
                hooks=hook_list,
                display_progress=True,
                test_interval=1)


if __name__ == '__main__':
    main()
