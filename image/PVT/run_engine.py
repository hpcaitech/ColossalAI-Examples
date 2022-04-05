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
import time
import datetime

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.optim import create_optimizer
from timm.utils import NativeScaler


def train_pvt_eigen():
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

    criterion = LabelSmoothingCrossEntropy()

    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    criterion = DistillationLoss(
        criterion, None, 'none', 0, 0
    )
    engine, train_dataloader, val_dataloader, _ = colossalai.initialize(model=model,
                                                                        optimizer=optimizer,
                                                                        criterion=criterion,
                                                                        train_dataloader=data_loader_train,
                                                                        test_dataloader=data_loader_val)
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(gpc.config.num_epochs):
        engine.train()
        for samples, targets in train_dataloader:
            samples = samples.to(device)
            targets = targets.to(device, non_blocking=True)
            if mixup_fn is not None:
                samples, targets = mixup_fn(samples, targets)
            with torch.cuda.amp.autocast():
                outputs = engine(samples)
                loss = engine.criterion(samples, outputs, targets)

            loss_value = loss.item()
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            engine.zero_grad()
            engine.backward(loss)
            engine.step()


        lr_scheduler.step()
        logger.info(f"Epoch {epoch} - train loss: {loss_value:.5}, lr: {lr_scheduler.get_last_lr()[0]:.5g}", ranks=[0])

        if args.output_dir:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            checkpoint_paths = [args.output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)

        engine.eval()
        criterion = torch.nn.CrossEntropyLoss()
        for samples, targets in val_dataloader:
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                outputs = engine(samples)
                loss = criterion(outputs, targets)
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))

        max_accuracy = max(max_accuracy, acc1)
        logger.info(f"Epoch {epoch} - eval loss: {loss:.5}, top1: {acc1:.5g}, top5: {acc5:.5g}, max_accuracy: {max_accuracy:.5g}", ranks=[0])

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))




if __name__ == '__main__':
    train_pvt_eigen()