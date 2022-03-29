import os
from typing import Callable, List, Optional, Type, Union

import colossalai
import colossalai.nn as col_nn
import torch
import torch.nn as nn
from colossalai.builder import build_pipeline_model
from colossalai.engine.schedule import (InterleavedPipelineSchedule,
                                        PipelineSchedule)
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.trainer import Trainer, hooks
from colossalai.utils import MultiTimer, get_dataloader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1


# Define model, modified from torchvision.models.resnet.ResNet
class ResNetClassifier(nn.Module):
    def __init__(self, expansion, num_classes) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * expansion, num_classes)

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def _make_layer(block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                norm_layer: nn.Module, dilation: int, inplanes: int, groups: int, base_width: int,
                stride: int = 1, dilate: bool = False) -> nn.Sequential:
    downsample = None
    previous_dilation = dilation
    if dilate:
        dilation *= stride
        stride = 1
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            conv1x1(inplanes, planes * block.expansion, stride),
            norm_layer(planes * block.expansion),
        )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample, groups,
                        base_width, previous_dilation, norm_layer))
    inplanes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(block(inplanes, planes, groups=groups,
                            base_width=base_width, dilation=dilation,
                            norm_layer=norm_layer))

    return nn.Sequential(*layers), dilation, inplanes


def resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    num_classes: int = 1000,
    zero_init_residual: bool = False,
    groups: int = 1,
    width_per_group: int = 64,
    replace_stride_with_dilation: Optional[List[bool]] = None,
    norm_layer: Optional[Callable[..., nn.Module]] = None
) -> None:
    if norm_layer is None:
        norm_layer = nn.BatchNorm2d

    inplanes = 64
    dilation = 1
    if replace_stride_with_dilation is None:
        # each element in the tuple indicates if we should replace
        # the 2x2 stride with a dilated convolution instead
        replace_stride_with_dilation = [False, False, False]
    if len(replace_stride_with_dilation) != 3:
        raise ValueError("replace_stride_with_dilation should be None "
                         "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
    groups = groups
    base_width = width_per_group
    conv = nn.Sequential(
        nn.Conv2d(3, inplanes, kernel_size=7, stride=2, padding=3,
                  bias=False),
        norm_layer(inplanes),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    layer1, dilation, inplanes = _make_layer(block, 64, layers[0], norm_layer, dilation, inplanes, groups, base_width)
    layer2, dilation, inplanes = _make_layer(block, 128, layers[1], norm_layer, dilation, inplanes, groups, base_width,
                                             stride=2, dilate=replace_stride_with_dilation[0])
    layer3, dilation, inplanes = _make_layer(block, 256, layers[2], norm_layer, dilation, inplanes, groups, base_width,
                                             stride=2, dilate=replace_stride_with_dilation[1])
    layer4, dilation, inplanes = _make_layer(block, 512, layers[3], norm_layer, dilation, inplanes, groups, base_width,
                                             stride=2, dilate=replace_stride_with_dilation[2])
    classifier = ResNetClassifier(block.expansion, num_classes)

    model = nn.Sequential(
        conv,
        layer1,
        layer2,
        layer3,
        layer4,
        classifier
    )

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    # Zero-initialize the last BN in each residual branch,
    # so that the residual branch starts with zeros, and each residual block behaves like an identity.
    # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
    if zero_init_residual:
        for m in model.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
            elif isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    return model


def resnet50():
    return resnet(Bottleneck, [3, 4, 6, 3])

# Process dataset


def build_cifar(batch_size):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = CIFAR10(root=os.environ['DATA'], train=True, download=True, transform=transform_train)
    test_dataset = CIFAR10(root=os.environ['DATA'], train=False, transform=transform_test)
    train_dataloader = get_dataloader(dataset=train_dataset, shuffle=True, batch_size=batch_size, pin_memory=True)
    test_dataloader = get_dataloader(dataset=test_dataset, batch_size=batch_size, pin_memory=True)
    return train_dataloader, test_dataloader


# Train

BATCH_SIZE = 64
NUM_EPOCHS = 2
NUM_CHUNKS = 1
CONFIG = dict(parallel=dict(pipeline=2))


def train():
    disable_existing_loggers()
    parser = colossalai.get_default_parser()
    args = parser.parse_args()
    colossalai.launch_from_torch(backend=args.backend, config=CONFIG)
    logger = get_dist_logger()

    # build model
    model = resnet50()
    model = build_pipeline_model(model, num_chunks=NUM_CHUNKS, verbose=True)

    # build criterion
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # build dataloader
    train_dataloader, test_dataloader = build_cifar(BATCH_SIZE)

    lr_scheduler = col_nn.lr_scheduler.LinearWarmupLR(optimizer, NUM_EPOCHS, warmup_steps=1)
    engine, train_dataloader, test_dataloader, lr_scheduler = colossalai.initialize(model, optimizer, criterion,
                                                                                    train_dataloader, test_dataloader, lr_scheduler)
    timer = MultiTimer()

    if NUM_CHUNKS == 1:
        schedule = PipelineSchedule(num_microbatches=4)
    else:
        schedule = InterleavedPipelineSchedule(num_microbatches=4, num_model_chunks=NUM_CHUNKS)

    trainer = Trainer(engine=engine, timer=timer, logger=logger, schedule=schedule)

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
