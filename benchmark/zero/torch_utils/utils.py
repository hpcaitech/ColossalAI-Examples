import os

import torch
from zero.common.utils import CONFIG, get_gpu_memory_mb, get_model_size, print_log
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP


def init_w_torch(builder):
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    host = os.environ['MASTER_ADDR']
    port = int(os.environ['MASTER_PORT'])
    init_process_group(rank=rank, world_size=world_size, init_method=f'tcp://{host}:{port}', backend='nccl')

    torch.cuda.set_device(rank)
    if CONFIG.get('gpu_mem_fraction', None) is not None:
        torch.cuda.set_per_process_memory_fraction(CONFIG['gpu_mem_fraction'])
        print_log(f'Set max GPU mem: {get_gpu_memory_mb() * CONFIG["gpu_mem_fraction"]:.2f} MB')

    build_data, build_model, build_loss, build_optimizer, build_scheduler = builder()

    train_data, test_data = build_data()

    model = build_model().to(rank)
    if 'numel' not in CONFIG['model']:
        CONFIG['model']['numel'] = get_model_size(model)
    model = DDP(model)

    criterion = build_loss()

    optimizer = build_optimizer(model.parameters())

    scaler = torch.cuda.amp.GradScaler(**CONFIG['fp16']) if 'fp16' in CONFIG else None

    lr_scheduler = build_scheduler(len(train_data), optimizer)

    return model, train_data, test_data, criterion, optimizer, scaler, lr_scheduler
