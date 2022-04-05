import os
import colossalai
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.nn.lr_scheduler import CosineAnnealingLR
from colossalai.nn.metric import Accuracy
from colossalai.utils import MultiTimer, get_dataloader
from colossalai.trainer import Trainer, hooks
import torch
from torch.autograd import Variable
from torch.utils.data import DistributedSampler, RandomSampler, SequentialSampler
from torch.utils.data import DataLoader # The directory of your dataset
import pvt
from timm.utils import accuracy, ModelEma
import utils
from datasets import build_dataset
from losses import DistillationLoss
import math
import sys
from pathlib import Path
import json
import time
import datetime

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma


def train_pvt():
    # DATASET_PATH = str(os.environ['DATA'])
    args_co = colossalai.get_default_parser().parse_args()
    device = torch.device("cuda")
    colossalai.launch_from_torch(config=args_co.config)
    logger = get_dist_logger()
    if hasattr(gpc.config, 'LOG_PATH'):
        if gpc.get_global_rank() == 0:
            log_path = gpc.config.LOG_PATH
            if not os.path.exists(log_path):
                os.mkdir(log_path)
            logger.log_to_file(log_path)

    args = gpc.config

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )
    model.to(device)

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model)
    loss_scaler = NativeScaler()
    lr_scheduler = CosineAnnealingLR(optimizer=optimizer, total_steps=gpc.config.num_epochs)

    criterion = torch.nn.CrossEntropyLoss()

    engine, train_dataloader, val_dataloader, _ = colossalai.initialize(model=model,
                                                                        optimizer=optimizer,
                                                                        criterion=criterion,
                                                                        train_dataloader=data_loader_train,
                                                                        test_dataloader=data_loader_val)
    # build a timer to measure time
    timer = MultiTimer()

    # create a trainer object
    trainer = Trainer(
        engine=engine,
        timer=timer,
        logger=logger
    )

    # define the hooks to attach to the trainer
    hook_list = [
        hooks.LossHook(),
        hooks.LRSchedulerHook(lr_scheduler=lr_scheduler, by_epoch=True),
        hooks.AccuracyHook(accuracy_func=Accuracy()),
        hooks.LogMetricByEpochHook(logger),
        hooks.LogMemoryByEpochHook(logger),
        hooks.LogTimingByEpochHook(timer, logger),

        # you can uncomment these lines if you wish to use them
        # hooks.TensorboardHook(log_dir='./tb_logs', ranks=[0]),
        # hooks.SaveCheckpointHook(checkpoint_dir='./ckpt')
    ]

    # start training
    trainer.fit(
        train_dataloader=train_dataloader,
        epochs=gpc.config.num_epochs,
        test_dataloader=val_dataloader,
        test_interval=1,
        hooks=hook_list,
        display_progress=True
    )


if __name__ == '__main__':
    train_pvt()