import glob
import os
import colossalai
import torch
import torch.nn as nn
import torch.nn.functional as F
import colossalai.utils as utils
from colossalai.utils import colo_set_process_memory_fraction, get_current_device, MultiTimer
from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn import Accuracy, CrossEntropyLoss
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from colossalai.nn.optimizer import HybridAdam
from colossalai.nn._ops import *
from colossalai.nn.parallel.layers import init_colo_module
from colossalai.nn.parallel import ZeroDDP
from colossalai.nn.parallel.data_parallel import ColoDDP
from colossalai.zero import ZeroOptimizer
from colossalai.trainer import Trainer, hooks
from colossalai.tensor import TensorSpec, ComputePattern, ParallelAction, DistSpecManager, distspec, ChunkManager
from colossalai.gemini.gemini_mgr import GeminiManager
from titans.dataloader.imagenet import build_dali_imagenet
from timm.models.vision_transformer import _create_vision_transformer
from tqdm import tqdm


def init_spec_func(model, tp_type):
    if tp_type == 'row':
        spec = TensorSpec(
            distspec.shard(gpc.get_group(ParallelMode.PARALLEL_1D), [-1],
                           [gpc.get_world_size(ParallelMode.PARALLEL_1D)]), ParallelAction(ComputePattern.TP1D))
        with DistSpecManager.no_grad():
            for n, p in model.named_parameters():
                if 'weight' in n and 'norm' not in n and 'patch_embed.proj.weight' not in n:
                    p.set_spec(spec)
    elif tp_type == 'col':
        spec = TensorSpec(
            distspec.shard(gpc.get_group(ParallelMode.PARALLEL_1D), [0],
                           [gpc.get_world_size(ParallelMode.PARALLEL_1D)]), ParallelAction(ComputePattern.TP1D))
        with DistSpecManager.no_grad():
            for n, p in model.named_parameters():
                if ('weight' in n or 'bias' in n) and 'norm' not in n and ('patch_embed.proj.weight' not in n
                                                                           and 'patch_embed.proj.bias' not in n):
                    p.set_spec(spec)
    else:
        raise NotImplemented


def train_imagenet():

    parser = colossalai.get_default_parser()
    parser.add_argument('--from_torch', default=True, action='store_true')
    args = parser.parse_args()
    colossalai.launch_from_torch(config=args.config)
    use_ddp = gpc.config.USE_DDP

    disable_existing_loggers()

    logger = get_dist_logger()
    if hasattr(gpc.config, 'LOG_PATH'):
        if gpc.get_global_rank() == 0:
            log_path = gpc.config.LOG_PATH
            if not os.path.exists(log_path):
                os.mkdir(log_path)
            logger.log_to_file(log_path)

    logger.info('Build data loader', ranks=[0])
    root = os.environ['DATA']
    train_dataloader, test_dataloader = build_dali_imagenet(root, rand_augment=False)

    logger.info('Build model', ranks=[0])

    model_kwargs = dict(img_size=gpc.config.IMG_SIZE,
                        patch_size=gpc.config.PATCH_SIZE,
                        embed_dim=gpc.config.HIDDEN_SIZE,
                        depth=gpc.config.DEPTH,
                        num_heads=gpc.config.NUM_HEADS,
                        mlp_ratio=gpc.config.MLP_RATIO,
                        num_classes=gpc.config.NUM_CLASSES,
                        weight_init='jax')

    use_chunk = True
    use_zero = True
    placement_policy = 'cuda'
    with ColoInitContext(device=get_current_device()):
        model = _create_vision_transformer('vit_small_patch16_224', pretrained=False, **model_kwargs)
    model = model.cuda().half()
    init_spec_func(model, gpc.config.TP_TYPE)
    chunk_size = ChunkManager.search_chunk_size(model, 8192, 8) if use_chunk else None
    chunk_manager = ChunkManager(chunk_size,
                                 enable_distributed_storage=use_zero,
                                 init_device=GeminiManager.get_default_device(placement_policy))
    gemini_manager = GeminiManager(placement_policy, chunk_manager)

    model = ZeroDDP(model, gemini_manager)

    logger.info('Build criterion, optimizer, lr_scheduler', ranks=[0])
    criterion = CrossEntropyLoss(label_smoothing=0.1)
    optimizer = HybridAdam(model.parameters(), lr=gpc.config.LEARNING_RATE, weight_decay=gpc.config.WEIGHT_DECAY)
    optimizer = ZeroOptimizer(optimizer, model, initial_scale=32)
    lr_scheduler = CosineAnnealingWarmupLR(optimizer=optimizer,
                                           total_steps=gpc.config.NUM_EPOCHS,
                                           warmup_steps=gpc.config.WARMUP_EPOCHS)

    start_epoch = 0
    for epoch in range(start_epoch, gpc.config.NUM_EPOCHS):
        model.train()
        for index, (x, y) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=False):
            x, y = x.cuda(), y.cuda()
            if use_ddp:
                model.zero_grad()
            else:
                optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            if use_ddp:
                model.backward(loss)
            else:
                loss.backward()
            optimizer.step()
            if index > 10:
                break
        logger.info(
            f"Finish Train Epoch [{epoch+1}/{gpc.config.NUM_EPOCHS}] loss: {loss.item():.3f} lr: {optimizer.state_dict()['param_groups'][0]['lr']}",
            ranks=[0])

        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for index, (x, y) in tqdm(enumerate(test_dataloader), total=len(test_dataloader), leave=False):
                x, y = x.cuda(), y.cuda()
                output = model(x)
                test_loss += F.nll_loss(output, y, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(y.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        logger.info(
            f"Finish Test Epoch [{epoch+1}/{gpc.config.NUM_EPOCHS}] loss: {test_loss:.3f} Accuracy: [{correct}/{len(test_dataloader.dataset)}]({correct/len(test_dataloader.dataset):.3f})",
            ranks=[0])


if __name__ == '__main__':
    train_imagenet()
