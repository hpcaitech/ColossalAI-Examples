import os

import colossalai
from colossalai.builder.pipeline import build_pipeline_model
from colossalai.context.parallel_mode import ParallelMode
import colossalai.nn as col_nn
from colossalai.nn.lr_scheduler.linear import LinearWarmupLR
import torch
import torch.nn as nn

from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.trainer import Trainer, hooks
from colossalai.utils import MultiTimer, get_dataloader, load_checkpoint
from torchvision import transforms
from torchvision.datasets import CIFAR10
from colossalai.core import global_context as gpc
from sequential_vit import sequential_vit


def vit_tiny_patch4_32(**kwargs):
    model_kwargs = dict(img_size=32, patch_size=4, num_classes=10, embed_dim=256, depth=6, num_heads=8, mlp_ratio=1, **kwargs)
    return sequential_vit(**model_kwargs)


def build_cifar(batch_size):
    transform_train = transforms.Compose([
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = CIFAR10(root=os.environ['DATA'], train=True, download=True, transform=transform_train)
    test_dataset = CIFAR10(root=os.environ['DATA'], train=False, transform=transform_test)
    train_dataloader = get_dataloader(dataset=train_dataset, shuffle=True, batch_size=batch_size, pin_memory=True)
    test_dataloader = get_dataloader(dataset=test_dataset, batch_size=batch_size, pin_memory=True)
    return train_dataloader, test_dataloader


# Train

BATCH_SIZE = 128
NUM_EPOCHS = 10
NUM_CHUNKS = 1
CONFIG = dict(NUM_MICRO_BATCHES=4, parallel=dict(tensor=dict(size=4, mode='2d')))#, model=dict(num_chunks=NUM_CHUNKS))
#CONFIG = dict(NUM_MICRO_BATCHES=4, parallel=dict(pipeline=2))
#, model=dict(num_chunks=NUM_CHUNKS))


def train():
    disable_existing_loggers()
    parser = colossalai.get_default_parser()
    args = parser.parse_args()
    colossalai.launch_from_torch(backend=args.backend, config=CONFIG)
    logger = get_dist_logger()
    # build model
    model = vit_tiny_patch4_32()
    #model = build_pipeline_model(model, NUM_CHUNKS)
    # build criterion
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)

    # build dataloader
    train_dataloader, test_dataloader = build_cifar(BATCH_SIZE)

    engine, train_dataloader, test_dataloader, _ = colossalai.initialize(model, optimizer, criterion,
                                                                         train_dataloader, test_dataloader)
    timer = MultiTimer()

    trainer = Trainer(engine=engine, timer=timer, logger=logger)

    hook_list = [
        hooks.LossHook(),
        hooks.AccuracyHook(col_nn.metric.Accuracy()),
        hooks.LogMetricByEpochHook(logger),
        hooks.SaveCheckpointHook(1, 'vit_cifar.pt', model)
    ]

    trainer.fit(train_dataloader=train_dataloader,
                epochs=NUM_EPOCHS,
                test_dataloader=test_dataloader,
                test_interval=1,
                hooks=hook_list,
                display_progress=True)


if __name__ == '__main__':
    train()
    gpc.destroy()
