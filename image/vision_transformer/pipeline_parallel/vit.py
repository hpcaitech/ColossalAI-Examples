import os
from collections import OrderedDict
from functools import partial

import colossalai
import colossalai.nn as col_nn
import torch
import torch.nn as nn
from colossalai.builder import build_pipeline_model
from colossalai.context import ParallelMode
from colossalai.engine.schedule import (InterleavedPipelineSchedule, PipelineSchedule)
from colossalai.core import global_context as gpc
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.trainer import Trainer, hooks
from colossalai.utils import MultiTimer, get_dataloader
from colossalai.utils.model.pipelinable import PipelinableContext
from titans.model.vit.vit import _create_vit_model
from torchvision import transforms
from torchvision.datasets import CIFAR10

# Process dataset


def build_cifar(batch_size):
    transform_train = transforms.Compose([
        transforms.RandomCrop(224, pad_if_needed=True),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = CIFAR10(root=os.environ['DATA'], train=True, download=True, transform=transform_train)
    test_dataset = CIFAR10(root=os.environ['DATA'], train=False, transform=transform_test)
    train_dataloader = get_dataloader(dataset=train_dataset, shuffle=True, batch_size=batch_size, pin_memory=True)
    test_dataloader = get_dataloader(dataset=test_dataset, batch_size=batch_size, pin_memory=True)
    return train_dataloader, test_dataloader


def train():
    args = colossalai.get_default_parser().parse_args()
    colossalai.launch_from_torch(config=args.config)
    logger = get_dist_logger()
    # build model
    model_kwargs = dict(img_size=gpc.config.IMG_SIZE,
                        patch_size=gpc.config.PATCH_SIZE,
                        dim=gpc.config.HIDDEN_SIZE,
                        depth=gpc.config.DEPTH,
                        num_heads=gpc.config.NUM_HEADS,
                        mlp_ratio=gpc.config.MLP_RATIO,
                        num_classes=gpc.config.NUM_CLASSES,
                        init_method='jax',
                        checkpoint=gpc.config.CHECKPOINT)

    pipelinable = PipelinableContext()
    with pipelinable:
        model = _create_vit_model(**model_kwargs)
    pipelinable.to_layer_list()
    pipelinable.load_policy("uniform")
    model = pipelinable.partition(1, gpc.pipeline_parallel_size, gpc.get_local_rank(ParallelMode.PIPELINE))

    # build criterion
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)

    # build dataloader
    train_dataloader, test_dataloader = build_cifar(gpc.config.BATCH_SIZE)

    engine, train_dataloader, test_dataloader, _ = colossalai.initialize(model, optimizer, criterion, train_dataloader,
                                                                         test_dataloader)
    timer = MultiTimer()

    trainer = Trainer(engine=engine, timer=timer, logger=logger)

    hook_list = [
        hooks.LossHook(),
        hooks.AccuracyHook(col_nn.metric.Accuracy()),
        hooks.LogMetricByEpochHook(logger),
    ]

    trainer.fit(train_dataloader=train_dataloader,
                epochs=2,
                test_dataloader=test_dataloader,
                test_interval=1,
                hooks=hook_list,
                display_progress=True)


if __name__ == '__main__':
    train()
