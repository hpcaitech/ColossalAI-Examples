#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import glob
import os

import colossalai

import torch
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.nn import CrossEntropyLoss
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from colossalai.utils import is_using_pp
from colossalai.engine.schedule import PipelineSchedule, NonPipelineSchedule
from dataloader import DaliDataloader
from model.vit import build_pipeline_vit
from model_zoo.vit.vit import _create_vit_model
from tqdm import tqdm


DATASET_PATH = os.environ['DATA']

TRAIN_RECS = DATASET_PATH + '/train/*'
VAL_RECS = DATASET_PATH + '/validation/*'
TRAIN_IDX = DATASET_PATH + '/idx_files/train/*'
VAL_IDX = DATASET_PATH + '/idx_files/validation/*'


def build_dali_train(batch_size):
    return DaliDataloader(
        sorted(glob.glob(TRAIN_RECS)),
        sorted(glob.glob(TRAIN_IDX)),
        batch_size=batch_size,
        shard_id=gpc.get_local_rank(ParallelMode.DATA),
        num_shards=gpc.get_world_size(ParallelMode.DATA),
        training=True,
        gpu_aug=False,
        cuda=False,
    )


def build_dali_test(batch_size):
    return DaliDataloader(
        sorted(glob.glob(VAL_RECS)),
        sorted(glob.glob(VAL_IDX)),
        batch_size=batch_size,
        shard_id=gpc.get_local_rank(ParallelMode.DATA),
        num_shards=gpc.get_world_size(ParallelMode.DATA),
        training=False,
        gpu_aug=False,
        cuda=False,
    )


def train_imagenet():
    args = colossalai.get_default_parser().parse_args()
    # standard launch
    colossalai.launch_from_slurm(config=args.config,
                                 host=args.host,
                                 port=29500)
    # if using torch distributed launcher
    # colossalai.launch_from_torch(config=args.config)

    logger = get_dist_logger()
    if hasattr(gpc.config, 'LOG_PATH'):
        if gpc.get_global_rank() == 0:
            log_path = gpc.config.LOG_PATH
            if not os.path.exists(log_path):
                os.mkdir(log_path)
            logger.log_to_file(log_path)

    use_pipeline = is_using_pp()

    # create model
    model_kwargs = dict(img_size=gpc.config.IMG_SIZE,
                        patch_size=gpc.config.PATCH_SIZE,
                        dim=gpc.config.HIDDEN_SIZE,
                        depth=gpc.config.DEPTH,
                        num_heads=gpc.config.NUM_HEADS,
                        mlp_ratio=gpc.config.MLP_RATIO,
                        num_classes=gpc.config.NUM_CLASSES,
                        init_method='jax',
                        checkpoint=gpc.config.CHECKPOINT)

    if use_pipeline:
        model = build_pipeline_vit(num_layers=model_kwargs['depth'], num_chunks=1, **model_kwargs)
    else:
        model = _create_vit_model(**model_kwargs)

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
    train_dataloader = build_dali_train(gpc.config.BATCH_SIZE)
    test_dataloader = build_dali_test(gpc.config.BATCH_SIZE)

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

    # create schedule
    schedule = None
    tensor_shape = getattr(gpc.config, 'TENSOR_SHAPE', None)
    if gpc.is_initialized(ParallelMode.PARALLEL_1D):
        scatter_gather = True
    else:
        scatter_gather = False

    data_iter = iter(train_dataloader)

    for epoch in range(gpc.config.NUM_EPOCHS):
        # training
        engine.train()

        if gpc.get_global_rank() == 0:
            description = 'Epoch {} / {}'.format(
                epoch,
                gpc.config.NUM_EPOCHS
            )
            progress = tqdm(range(len(train_dataloader)), desc=description)
        else:
            progress = range(len(train_dataloader))
        for _ in progress:
            engine.zero_grad()
            engine.execute_schedule(data_iter, return_output_label=False)
            engine.step()
            lr_scheduler.step()


if __name__ == '__main__':
    train_imagenet()
