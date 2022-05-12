import os
from typing import Callable, List, Optional, Type, Union
import torch
import torch.nn as nn
import colossalai
import colossalai.nn as col_nn

from colossalai.core import global_context as gpc
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.trainer import Trainer, hooks
from colossalai.utils import MultiTimer, get_dataloader
from colossalai.context import ParallelMode
from colossalai.utils.model.pipelinable import PipelinableContext

from titans.dataloader.cifar10 import build_cifar
from torchvision.models import resnet50
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1

# Train

BATCH_SIZE = 64
NUM_EPOCHS = 2
NUM_CHUNKS = 1
CONFIG = dict(NUM_MICRO_BATCHES=4, parallel=dict(pipeline=2))


def train():
    disable_existing_loggers()
    parser = colossalai.get_default_parser()
    args = parser.parse_args()
    colossalai.launch_from_torch(backend=args.backend, config=CONFIG)
    logger = get_dist_logger()
    pipelinable = PipelinableContext()

    # build model
    with pipelinable:
        model = resnet50()

    exec_seq = [
        'conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool',
        (lambda x: torch.flatten(x, 1), "behind"), 'fc'
    ]
    pipelinable.to_layer_list(exec_seq)
    model = pipelinable.partition(NUM_CHUNKS, gpc.pipeline_parallel_size, gpc.get_local_rank(ParallelMode.PIPELINE))

    # build criterion
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # build dataloader
    root = os.environ.get('DATA', './data')
    train_dataloader, test_dataloader = build_cifar(BATCH_SIZE, root, padding=4, crop=32, resize=32)

    lr_scheduler = col_nn.lr_scheduler.LinearWarmupLR(optimizer, NUM_EPOCHS, warmup_steps=1)
    engine, train_dataloader, test_dataloader, lr_scheduler = colossalai.initialize(model, optimizer, criterion,
                                                                                    train_dataloader, test_dataloader,
                                                                                    lr_scheduler)
    timer = MultiTimer()

    trainer = Trainer(engine=engine, timer=timer, logger=logger)

    hook_list = [
        hooks.LossHook(),
        hooks.AccuracyHook(col_nn.metric.Accuracy()),
        hooks.LogMetricByEpochHook(logger),
        hooks.LRSchedulerHook(lr_scheduler, by_epoch=True)
    ]

    trainer.fit(train_dataloader=train_dataloader,
                epochs=NUM_EPOCHS,
                test_dataloader=test_dataloader,
                test_interval=1,
                hooks=hook_list,
                display_progress=True)


if __name__ == '__main__':
    train()
