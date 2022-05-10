import os

import torch
from zero.common.utils import CONFIG, get_gpu_memory_mb, print_log
from torch.distributed import init_process_group


def init_w_ps(builder):
    from patrickstar.runtime import initialize_engine

    config = CONFIG.copy()

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    host = os.environ['MASTER_ADDR']
    port = int(os.environ['MASTER_PORT'])
    init_process_group(rank=rank, world_size=world_size, init_method=f'tcp://{host}:{port}', backend='nccl')

    torch.cuda.set_device(rank)
    if CONFIG.get('gpu_mem_fraction', None) is not None:
        torch.cuda.set_per_process_memory_fraction(CONFIG['gpu_mem_fraction'])
        print_log(f'Set max GPU mem: {get_gpu_memory_mb() * CONFIG["gpu_mem_fraction"]:.2f} MB')

    build_data, build_model, build_loss, _, build_scheduler = builder()

    train_data, test_data = build_data()

    criterion = build_loss()

    model, optimizer = initialize_engine(model_func=build_model, local_rank=rank, config=config)

    lr_scheduler = build_scheduler(len(train_data), optimizer)

    return model, train_data, test_data, criterion, optimizer, None, lr_scheduler
