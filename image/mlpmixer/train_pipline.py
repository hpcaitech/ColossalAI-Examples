import torch
import torch.nn as nn

import os
from collections import OrderedDict
from functools import partial

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

import glob
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.nn import Accuracy, CrossEntropyLoss
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from colossalai.trainer import Trainer, hooks
from colossalai.utils import MultiTimer, is_using_pp

from colossalai.utils.model.pipelinable import PipelinableContext



class MlpBlock(nn.Module):
    def __init__(self, hidden_dim, mlp_dim):
        super(MlpBlock, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class MixerBlock(nn.Module):
    def __init__(self, num_tokens, hidden_dim, tokens_mlp_dim, channels_mlp_dim):
        super(MixerBlock, self).__init__()
        self.ln_token = nn.LayerNorm(hidden_dim)
        self.token_mix = MlpBlock(num_tokens, tokens_mlp_dim)
        self.ln_channel = nn.LayerNorm(hidden_dim)
        self.channel_mix = MlpBlock(hidden_dim, channels_mlp_dim)

    def forward(self, x):
        out = self.ln_token(x).transpose(1, 2)
        x = x + self.token_mix(out).transpose(1, 2)
        out = self.ln_channel(x)
        x = x + self.channel_mix(out)
        return x


# class MlpMixer(nn.Module):
#     def __init__(self, num_classes, num_blocks, patch_size, hidden_dim, tokens_mlp_dim, channels_mlp_dim, image_size=224):
#         super(MlpMixer, self).__init__()
#         num_tokens = (image_size // patch_size)**2
#
#         self.patch_emb = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size, bias=False)
#
#         self.mlp = nn.Sequential(*[MixerBlock(num_tokens, hidden_dim, tokens_mlp_dim, channels_mlp_dim) for _ in range(num_blocks)])
#         self.ln = nn.LayerNorm(hidden_dim)
#         self.fc = nn.Linear(hidden_dim, num_classes)
#
#         self.mlpm = nn.Sequential(
#             self.patch_emb,
#             data_flatten(),
#             self.mlp,
#             data_mean(),
#             self.ln,
#             self.fc)
#
#     def forward(self, x):
#         return self.mlpm(x)


class data_flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.flatten(2).transpose(1, 2)

class data_mean(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.mean(dim=1)

def MlpMixer(num_classes, num_blocks, patch_size, hidden_dim, tokens_mlp_dim, channels_mlp_dim, image_size=224):
    num_tokens = (image_size // patch_size) ** 2
    patch_emb = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size, bias=False)

    mlp = nn.Sequential(
        *[MixerBlock(num_tokens, hidden_dim, tokens_mlp_dim, channels_mlp_dim) for _ in range(num_blocks)])
    ln = nn.LayerNorm(hidden_dim)
    fc = nn.Linear(hidden_dim, num_classes)

    return nn.Sequential(patch_emb,data_flatten(),mlp,ln, data_mean(), fc)

# def data_flatten(x):
#     return x.flatten(2).transpose(1, 2)
#
# def data_mean(x):
#     return x.mean(dim=1)


# def mixer_s32(num_classes=10, image_size=32, patch_size=4,**kwargs):
#     return MlpMixer(num_classes, 8, patch_size, 512, 256, 2048, image_size)

def mixer_s32(num_classes=1000, image_size=224, patch_size=32,**kwargs):
    return MlpMixer(num_classes, 8, patch_size, 512, 256, 2048, image_size, **kwargs)


def build_cifar(batch_size):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, pad_if_needed=True),
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

BATCH_SIZE = 128
NUM_EPOCHS = 2
NUM_CHUNKS = 1
CONFIG = dict(NUM_MICRO_BATCHES=4, parallel=dict(pipeline=2))
num_classes = 10
image_size = 32
patch_size = 4



def train():
    # initialize distributed setting
    parser = colossalai.get_default_parser()
    args = parser.parse_args()

    # launch from torch
    colossalai.launch_from_torch(config=args.config)

    # get logger
    logger = get_dist_logger()
    logger.info("initialized distributed environment", ranks=[0])

    if hasattr(gpc.config, 'LOG_PATH'):
        if gpc.get_global_rank() == 0:
            log_path = gpc.config.LOG_PATH
            if not os.path.exists(log_path):
                os.mkdir(log_path)
            logger.log_to_file(log_path)

    use_pipeline = is_using_pp()

    # pipelinable = PipelinableContext()
    # with pipelinable:
    #     model = mixer_s32(num_classes,image_size,patch_size)
    # pipelinable.to_layer_list()
    # pipelinable.load_policy("uniform")
    # model = pipelinable.partition(1, gpc.pipeline_parallel_size, gpc.get_local_rank(ParallelMode.PIPELINE))

    if use_pipeline:
        pipelinable = PipelinableContext()
        with pipelinable:
            model = mixer_s32(num_classes,image_size,patch_size)
        pipelinable.to_layer_list()
        pipelinable.load_policy("uniform")
        model = pipelinable.partition(1, gpc.pipeline_parallel_size, gpc.get_local_rank(ParallelMode.PIPELINE))
    else:
        model = mixer_s32(num_classes,image_size,patch_size)


    # count number of parameters
    total_numel = 0
    for p in model.parameters():
        total_numel += p.numel()
    if not gpc.is_initialized(ParallelMode.PIPELINE):
        pipeline_stage = 0
    else:
        pipeline_stage = gpc.get_local_rank(ParallelMode.PIPELINE)
    logger.info(f"number of parameters: {total_numel} on pipeline stage {pipeline_stage}")

    # craete dataloaders
 
    train_dataloader, test_dataloader = build_cifar(BATCH_SIZE)

    # create loss function
    criterion = CrossEntropyLoss(label_smoothing=0.1)

    # create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=gpc.config.LEARNING_RATE, weight_decay=gpc.config.WEIGHT_DECAY)

    # create lr scheduler
    lr_scheduler = CosineAnnealingWarmupLR(optimizer=optimizer,
                                           total_steps=gpc.config.NUM_EPOCHS,
                                           warmup_steps=gpc.config.WARMUP_EPOCHS)

    # intiailize
    engine, train_dataloader, test_dataloader, _ = colossalai.initialize(model=model,
                                                                         optimizer=optimizer,
                                                                         criterion=criterion,
                                                                         train_dataloader=train_dataloader,
                                                                         test_dataloader=test_dataloader)

    logger.info("Engine is built", ranks=[0])

    # create timer
    timer = MultiTimer()

    # create trainer
    trainer = Trainer(engine=engine, logger=logger, timer=timer)
    logger.info("Trainer is built", ranks=[0])

    # create a list of useful hooks
    hook_list = [
        hooks.LogMetricByEpochHook(logger=logger),
        hooks.LogMetricByStepHook(),
        hooks.AccuracyHook(accuracy_func=Accuracy()),
        hooks.LossHook(),
        hooks.ThroughputHook(),
        hooks.LRSchedulerHook(lr_scheduler=lr_scheduler, by_epoch=True)
    ]

    # start training
    logger.info("Train start", ranks=[0])
    trainer.fit(
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        epochs=gpc.config.NUM_EPOCHS,
        hooks=hook_list,
        display_progress=True,
        test_interval=1,
    )


if __name__ == '__main__':
    train()
