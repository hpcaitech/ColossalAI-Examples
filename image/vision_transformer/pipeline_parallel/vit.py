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
from timm.models import vision_transformer as vit
from torchvision import transforms
from torchvision.datasets import CIFAR10

# Define model


class ViTEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, embed_layer=vit.PatchEmbed, drop_rate=0., distilled=False):
        super().__init__()
        self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.init_weights()

    def forward(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        return x

    def init_weights(self):
        vit.trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            vit.trunc_normal_(self.dist_token, std=.02)
        vit.trunc_normal_(self.cls_token, std=.02)
        self.apply(vit._init_vit_weights)


class ViTHead(nn.Module):
    def __init__(self, embed_dim=768, num_classes=1000, norm_layer=None, distilled=False, representation_size=None):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(embed_dim)
        self.num_classes = num_classes
        self.distilled = distilled
        self.num_features = embed_dim
        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()
        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.init_weights()

    def forward(self, x):
        x = self.norm(x)
        if self.distilled:
            x, x_dist = self.head(x[:, 0]), self.head_dist(x[:, 1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.pre_logits(x[:, 0])
            x = self.head(x)
        return x

    def init_weights(self):
        self.apply(vit._init_vit_weights)


def sequential_vit(img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                   num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                   drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=vit.PatchEmbed, norm_layer=None,
                   act_layer=None):
    norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
    act_layer = act_layer or nn.GELU
    embedding = ViTEmbedding(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                             embed_dim=embed_dim, embed_layer=embed_layer, drop_rate=drop_rate, distilled=distilled)
    dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
    blocks = [vit.Block(
        dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
        attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
        for i in range(depth)]
    for block in blocks:
        block.apply(vit._init_vit_weights)
    head = ViTHead(embed_dim=embed_dim, num_classes=num_classes, norm_layer=norm_layer,
                   distilled=distilled, representation_size=representation_size)
    return nn.Sequential(embedding, *blocks, head)


def vit_large_patch16_224(**kwargs):
    model_kwargs = dict(embed_dim=1024, depth=24, num_heads=16, **kwargs)
    return sequential_vit(**model_kwargs)


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


# Train

BATCH_SIZE = 128
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
    model = vit_large_patch16_224()
    model = build_pipeline_model(model, num_chunks=NUM_CHUNKS, verbose=True)

    # build criterion
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)

    # build dataloader
    train_dataloader, test_dataloader = build_cifar(BATCH_SIZE)

    engine, train_dataloader, test_dataloader, _ = colossalai.initialize(model, optimizer, criterion,
                                                                         train_dataloader, test_dataloader)
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
    ]

    trainer.fit(train_dataloader=train_dataloader,
                epochs=NUM_EPOCHS,
                test_dataloader=test_dataloader,
                test_interval=1,
                hooks=hook_list,
                display_progress=True)


if __name__ == '__main__':
    train()
