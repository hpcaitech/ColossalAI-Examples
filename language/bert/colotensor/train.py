import os

import colossalai
import colossalai.utils as utils
import torch
import torch.nn as nn
from torch.distributed import all_reduce, get_rank, get_world_size, is_initialized
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn import LinearWarmupLR
from colossalai.utils import colo_set_process_memory_fraction, get_current_device, MultiTimer
from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.nn._ops import *
from colossalai.nn.parallel.layers import init_colo_module
from colossalai.tensor import TensorSpec, ComputePattern, ComputeSpec, ChunkManager
from colossalai.gemini.gemini_mgr import GeminiManager
from colossalai.nn.parallel import ZeroDDP
from colossalai.nn.optimizer import HybridAdam
from colossalai.zero import ZeroOptimizer

from language.bert.colotensor.dataset import build_data
from language.bert.colotensor.model import build_model
from language.bert.colotensor.utils import AsyncMemoryMonitor, train, test, calc_local_model_size


def main():
    parser = colossalai.get_default_parser()
    parser.add_argument('--from_torch', default=True, action='store_true')
    args = parser.parse_args()
    disable_existing_loggers()
    colossalai.launch_from_torch(config=args.config)

    logger = get_dist_logger()

    logger.info('Build data loader', ranks=[0])
    train_dataloader, test_dataloader = build_data(
        dataset_path=os.environ["DATA"],
        tokenizer_path=os.environ["TOKENIZER"],
        seq_len=gpc.config.SEQ_LENGTH,
        batch_size=gpc.config.BATCH_SIZE,
    )

    logger.info('Build model', ranks=[0])
    use_zero = True

    with ColoInitContext(device=get_current_device()):
        model = build_model()
        
    compute_spec = ComputeSpec(ComputePattern.TP1D)
    init_colo_module(model, compute_spec, recursive=True, mode='col')

    use_chunk = True
    placement_policy = 'cuda'
    optimizer_class = HybridAdam
    lr = gpc.config.LR
    if use_zero:
        chunk_size = ChunkManager.search_chunk_size(model, 8192, 8) if use_chunk else None
        chunk_manager = ChunkManager(chunk_size,
                                    enable_distributed_storage=use_zero,
                                    init_device=GeminiManager.get_default_device(placement_policy))
        gemini_manager = GeminiManager(placement_policy, chunk_manager)
        model = ZeroDDP(model, gemini_manager)
        optimizer = optimizer_class(model.parameters(), lr=lr, weight_decay=1e-2, adamw_mode=True)
        optimizer = ZeroOptimizer(optimizer, model, initial_scale=32)
    else:
        optimizer = optimizer_class(model.parameters(), lr=lr, weight_decay=1e-2, adamw_mode=True)

    numel = calc_local_model_size(model)
    criterion = nn.CrossEntropyLoss()
    logger.info('Build optimizer', ranks=[0])

    total_steps = gpc.config.NUM_EPOCHS * len(train_dataloader)
    warmup_proportion = gpc.config.WARMUP_PROPORTION
    lr_scheduler = LinearWarmupLR(optimizer, total_steps=total_steps, warmup_steps=total_steps * warmup_proportion)

    global_batch_size = gpc.config.BATCH_SIZE * \
        gpc.get_world_size(ParallelMode.DATA) * getattr(gpc.config, "gradient_accumulation", 1)
    logger.info(f'Init done, global batch size = {global_batch_size}', ranks=[0])

    rank = get_rank()
    world_size = get_world_size()
    mem_monitor = AsyncMemoryMonitor(rank)
    for epoch in range(gpc.config.NUM_EPOCHS):
        train(epoch, rank, world_size, train_dataloader, model, criterion, optimizer, lr_scheduler, mem_monitor, numel, use_zero)
        test(epoch, rank, world_size, test_dataloader, model, criterion, mem_monitor, numel)

if __name__ == '__main__':
    main()