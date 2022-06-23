import os

import colossalai
import torch
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.utils import get_dataloader
from colossalai.nn.lr_scheduler import LinearWarmupLR
from colossalai.nn.metric import Accuracy
from colossalai.trainer import Trainer, hooks
from timm.models import vit_base_patch16_224

from titans.dataloader.cifar10 import build_cifar


def main():
    # initialize distributed setting
    parser = colossalai.get_default_parser()
    args = parser.parse_args()
    disable_existing_loggers()

    # launch from torch
    colossalai.launch_from_torch(config=args.config)

    # get logger
    logger = get_dist_logger()
    logger.info("initialized distributed environment", ranks=[0])

    # build model
    model = vit_base_patch16_224(drop_rate=0.1, num_classes=10)

    # build dataloader
    root = os.environ.get('DATA', './data')
    train_dataloader, test_dataloader = build_cifar(gpc.config.BATCH_SIZE, root, pad_if_needed=True)

    # build optimizer
    optimizer = colossalai.nn.Lamb(model.parameters(), lr=1.8e-2, weight_decay=0.1)

    # build loss
    criterion = torch.nn.CrossEntropyLoss()

    # lr_scheduler
    lr_scheduler = LinearWarmupLR(optimizer, warmup_steps=50, total_steps=gpc.config.NUM_EPOCHS)

    engine, train_dataloader, test_dataloader, _ = colossalai.initialize(model, optimizer, criterion, train_dataloader,
                                                                         test_dataloader)
    logger.info("initialized colossalai components", ranks=[0])

    # build trainer
    trainer = Trainer(engine=engine, logger=logger)

    # build hooks
    hook_list = [
        hooks.LossHook(),
        hooks.AccuracyHook(accuracy_func=Accuracy()),
        hooks.LogMetricByEpochHook(logger),
        hooks.LRSchedulerHook(lr_scheduler, by_epoch=True),
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
