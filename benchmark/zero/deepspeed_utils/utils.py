import torch
from zero.common.utils import CONFIG, get_gpu_memory_mb, print_log


def init_w_ds(builder):
    import deepspeed

    config = CONFIG.copy()

    deepspeed.init_distributed()

    if CONFIG.get('gpu_mem_fraction', None) is not None:
        torch.cuda.set_per_process_memory_fraction(CONFIG['gpu_mem_fraction'])
        print_log(f'Set max GPU mem: {get_gpu_memory_mb() * CONFIG["gpu_mem_fraction"]:.2f} MB')

    build_data, build_model, build_loss, build_optimizer, build_scheduler = builder()

    train_data, test_data = build_data()

    with deepspeed.zero.Init(config_dict_or_path=config):
        model = build_model()

    criterion = build_loss()

    optimizer = build_optimizer(model.parameters())

    lr_scheduler = build_scheduler(len(train_data), optimizer)

    model, optimizer, _, lr_scheduler = deepspeed.initialize(model=model,
                                                             optimizer=optimizer,
                                                             lr_scheduler=lr_scheduler,
                                                             config=config)

    return model, train_data, test_data, criterion, optimizer, None, lr_scheduler
