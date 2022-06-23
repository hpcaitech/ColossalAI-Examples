import torch
from torch.cuda import max_memory_allocated, reset_peak_memory_stats
from torch.distributed import get_rank
from zero.common.utils import (CONFIG, get_gpu_memory_mb, get_model_size, print_log)


def init_w_col(builder):
    import colossalai
    from colossalai.core import global_context as gpc
    from colossalai.logging import disable_existing_loggers
    from colossalai.nn.optimizer import CPUAdam
    from colossalai.zero.init_ctx import ZeroInitContext
    from colossalai.zero.shard_utils import (BucketTensorShardStrategy, TensorShardStrategy)
    from colossalai.zero.sharded_model import ShardedModel, ShardedModelV2
    from colossalai.zero.sharded_optim import ShardedOptimizerV2

    disable_existing_loggers()
    colossalai.launch_from_torch(config=CONFIG)

    if CONFIG.get('gpu_mem_fraction', None) is not None:
        torch.cuda.set_per_process_memory_fraction(CONFIG['gpu_mem_fraction'])
        print_log(f'Set max GPU mem: {get_gpu_memory_mb() * CONFIG["gpu_mem_fraction"]:.2f} MB')

    build_data, build_model, build_loss, optimizer_class, build_scheduler = builder()

    print_log('Building data')
    train_data, test_data = build_data()

    use_v2 = gpc.config.zero.pop('version', 2) == 2

    cpu_offload = gpc.config.zero.offload_config.device == 'cpu'

    rank = get_rank()
    reset_peak_memory_stats(rank)

    print_log('Building model')
    if use_v2:
        shard_strategy = TensorShardStrategy()
        model_numel = torch.zeros(1, dtype=torch.long)
        with ZeroInitContext(target_device=torch.cuda.current_device(),
                             shard_strategy=shard_strategy,
                             shard_param=True,
                             model_numel_tensor=model_numel,
                             rm_torch_payload_on_the_fly=True):
            model = build_model()
        model = ShardedModelV2(model, shard_strategy, **gpc.config.zero)
        if 'numel' not in CONFIG['model']:
            CONFIG['model']['numel'] = model_numel.item()
        print_log(f'model numel: {model_numel.item()}')
    else:
        model = build_model()
        if 'numel' not in CONFIG['model']:
            CONFIG['model']['numel'] = get_model_size(model)
        model = ShardedModel(model, **gpc.config.zero)

    criterion = build_loss()

    print_log(f'Peak Memory = {max_memory_allocated(rank) / (1024 * 1024)} M')
    reset_peak_memory_stats(rank)

    optimizer_kwargs = {}
    if cpu_offload:
        optimizer_class = CPUAdam
        optimizer_kwargs = {
            'lr': CONFIG['hyperparameter']['learning_rate'],
            'weight_decay': CONFIG['hyperparameter']['weight_decay']
        }

    if use_v2:
        optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
        optimizer = ShardedOptimizerV2(model, optimizer, **gpc.config.get('fp16', dict()), cpu_offload=cpu_offload)
    else:
        optimizer = optimizer_class(model.parameters())

    lr_scheduler = build_scheduler(len(train_data), optimizer)
    print_log(f'Peak Memory = {max_memory_allocated(rank) / (1024 * 1024)} M')

    return model, train_data, test_data, criterion, optimizer, None, lr_scheduler
