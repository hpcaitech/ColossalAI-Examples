import colossalai
import torch
import os

import colossalai.nn as col_nn
from colossalai.utils import get_dataloader,  MultiTimer
from colossalai.logging import get_dist_logger
from colossalai.core import global_context as gpc
from torch.nn.modules import CrossEntropyLoss
from torchvision import transforms
from torchvision.datasets import CIFAR10
from colossalai.trainer import Trainer, hooks
from model_zoo.vit import vit_tiny_patch4_32

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
    test_dataset = CIFAR10(root=os.environ['DATA'], train=False, download=True, transform=transform_test)
    train_dataloader = get_dataloader(dataset=train_dataset, shuffle=True, batch_size=batch_size, pin_memory=True)
    test_dataloader = get_dataloader(dataset=test_dataset, batch_size=batch_size, pin_memory=True)
    return train_dataloader, test_dataloader


BATCH_SIZE = 128
NUM_EPOCHS = 10
CONFIG = dict()


def train():
    args = colossalai.get_default_parser().parse_args()
    colossalai.launch_from_torch(backend=args.backend, config=CONFIG)
    
    logger = get_dist_logger()
    model = vit_tiny_patch4_32()
    criterion = CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
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
    