import os

from tqdm import tqdm
import time

import colossalai
import colossalai.utils as utils
import torch
import torch.nn as nn
from torch.distributed import all_reduce, get_rank, get_world_size, is_initialized
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn import LinearWarmupLR
from colossalai.trainer import Trainer, hooks
from colossalai.utils import colo_set_process_memory_fraction, get_current_device, MultiTimer
from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.nn._ops import *
from colossalai.nn.parallel.layers import init_colo_module
from colossalai.tensor import TensorSpec, ComputePattern, ParallelAction, ChunkManager
from colossalai.gemini.gemini_mgr import GeminiManager
from colossalai.nn.parallel import ColoDDPV2

from language.bert.colotensor.dataset import build_data
from language.bert.colotensor.model import build_model

from concurrent.futures import ThreadPoolExecutor
class AsyncMemoryMonitor:

    def __init__(self, rank, power=3, save_to_disk=True):
        """
        Adapted from https://github.com/Tencent/PatrickStar/blob/master/patrickstar/core/memtracer/memtracer.py.
        An Async Mem Monitor runing during computing.
        Sampling GPU memory usage of the current GPU dev
        at interval of 1/(10**power) sec.
        """
        self.keep_measuring = False
        device = torch.cuda.current_device()
        self.executor = ThreadPoolExecutor(max_workers=1, initializer=lambda: torch.cuda.set_device(device))
        self.monitor_thread = None
        self.interval = 1 / (10**power)
        self.rank = rank
        self.file = os.path.join(gpc.config.LOG_DIR, f'memory_rank_{rank}.log') if save_to_disk else None

    def set_interval(self, power: int):
        self.interval = 1 / (10**power)

    def start(self):
        self.keep_measuring = True
        torch.cuda.reset_peak_memory_stats(self.rank)
        self.monitor_thread = self.executor.submit(self._measure_usage)

    def finish(self):
        if self.keep_measuring is False:
            return 0
        self.keep_measuring = False
        gpu_usage = self.monitor_thread.result()
        self.monitor_thread = None
        if self.file is not None:
            with open(self.file, 'a') as f:
                f.writelines(list(map(lambda x: str(x) + '\n', gpu_usage)))
        return gpu_usage

    def _measure_usage(self):
        gpu_usage = list()
        while self.keep_measuring:
            gpu_usage.append(torch.cuda.max_memory_allocated(self.rank) / (1024 * 1024))  # MB
            torch.cuda.reset_peak_memory_stats(self.rank)
            time.sleep(self.interval)

        return gpu_usage

def print_log(msg):
    msg = f'{time.asctime()} > {msg}'
    rank = get_rank() if is_initialized() else 0
    log_file = os.path.join(gpc.config.LOG_DIR, f'training_rank_{rank}.log')
    with open(log_file, 'a') as f:
        f.write(msg + '\n')
    if rank == 0:
        print(msg)

def get_tflops(iter_time: float, num_tokens: int, numel) -> float:
    flops = numel * num_tokens * 2.0 * 4.0
    return (flops / 1e12) / (iter_time + 1e-12)

def _train(epoch, rank, world_size, train_dataloader, model, criterion, optimizer, lr_scheduler, mem_monitor, numel):
    model.train()

    num_steps = len(train_dataloader)
    progress = range(num_steps)

    if rank == 0:
        progress = tqdm(progress, desc=f"[Epoch {epoch} / Train]")

    train_loss = torch.zeros(()).to(torch.float).to(rank)
    used_time = 0.
    num_steps = 0
    num_samples = torch.zeros(()).to(torch.int).to(rank)
    num_tokens = torch.zeros(()).to(torch.int).to(rank)

    data_iter = iter(train_dataloader)

    if mem_monitor is not None:
        mem_monitor.start()

    for _ in progress:
        fwd_start = time.time()

        optimizer.zero_grad()

        batch = next(data_iter)

        labels = batch.pop('labels')
        batch_size = None
        batch_tokens = None
        if isinstance(labels, torch.Tensor):
            labels = labels.to(rank)
            batch_size = labels.size(0)
            batch_tokens = labels.numel()
        else:
            for k, v in labels.items():
                labels[k] = v.to(rank)
                if batch_size is None:
                    batch_size = v.size(0)
                if batch_tokens is None:
                    batch_tokens = v.numel()

        for k, v in batch.items():
            batch[k] = v.to(rank)

        outputs = model(**batch)

        shift_logits = outputs[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        train_loss += loss

        fwd_end = time.time()

        bwd_start = time.time()

        optimizer.backward(loss)
        optimizer.step()
        lr_scheduler.step()

        bwd_end = time.time()

        num_steps += 1
        num_samples += batch_size
        num_tokens += batch_tokens

        fwd_time = fwd_end - fwd_start
        bwd_time = bwd_end - bwd_start
        batch_time = fwd_time + bwd_time
        used_time += batch_time

        if rank == 0:
            progress.set_postfix(loss=loss.item(),
                                 lr=lr_scheduler.get_last_lr()[0],
                                 time_forward=fwd_time,
                                 time_backward=bwd_time,
                                 throughput=batch_size * world_size / (batch_time + 1e-12),
                                 tflops=get_tflops(batch_time, batch_tokens * world_size, numel))

    peak_mem = None
    if mem_monitor is not None:
        peak_mem = max(mem_monitor.finish())

    all_reduce(train_loss)
    all_reduce(num_samples)
    all_reduce(num_tokens)

    msg = f'[Epoch {epoch} / Train]: Loss = {train_loss.item() / (world_size * num_steps):.3f}'
    msg += f' | Throughput = {num_samples.item() / (used_time + 1e-12):.3f} samples/sec'
    msg += f' | TFLOPS = {get_tflops(used_time, num_tokens.item(), numel):.3f}'
    if peak_mem is not None:
        msg += f' | Peak memory = {peak_mem / 1024:.3f} GB.'
    print_log(msg)

def calc_local_model_size(model: torch.nn.Module):
    numel_per_device = 0
    for p in model.parameters():
        numel_per_device += p.numel()
    return numel_per_device


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
    use_zero = hasattr(gpc.config, 'zero')

    with ColoInitContext(device=get_current_device()):
        model = build_model()
        
    parallel_action = ParallelAction(ComputePattern.TP1D)
    #init_colo_module(model, parallel_action, recursive=True, mode='col')

    # TODO(jzy) Add ZERO
    use_chunk = False
    placement_policy = 'cuda'
    if use_zero:
        chunk_size = ChunkManager.search_chunk_size(model, 8192, 8) if use_chunk else None
        chunk_manager = ChunkManager(chunk_size,
                                    enable_distributed_storage=use_zero,
                                    init_device=GeminiManager.get_default_device(placement_policy))
        gemini_manager = GeminiManager(placement_policy, chunk_manager)
        model = ColoDDPV2(model, gemini_manager)

    numel = calc_local_model_size(model)

    criterion = nn.CrossEntropyLoss()

    logger.info('Build optimizer', ranks=[0])
    optimizer_class = torch.optim.AdamW
    optimizer = optimizer_class(model.parameters(), lr=5e-5, weight_decay=1e-2)

    lr_scheduler = LinearWarmupLR(optimizer, total_steps=gpc.config.NUM_EPOCHS * len(train_dataloader), warmup_steps=gpc.config.WARMUP_EPOCHS)

    engine, train_dataloader, _, lr_scheduler = colossalai.initialize(model,
                                                                      optimizer,
                                                                      criterion,
                                                                      train_dataloader=train_dataloader,
                                                                      lr_scheduler=lr_scheduler)
    global_batch_size = gpc.config.BATCH_SIZE * \
        gpc.get_world_size(ParallelMode.DATA) * getattr(gpc.config, "gradient_accumulation", 1)
    logger.info(f'Init done, global batch size = {global_batch_size}', ranks=[0])

    rank = get_rank()
    world_size = get_world_size()
    mem_monitor = AsyncMemoryMonitor(rank)
    optimizer = engine
    for epoch in range(gpc.config.NUM_EPOCHS):
        _train(epoch, rank, world_size, train_dataloader, model, criterion, optimizer, lr_scheduler, mem_monitor, numel)

    # def data_process_func(batch_data):
    #     data = {
    #         'input_ids': batch_data['input_ids'],
    #         'token_type_ids': batch_data['token_type_ids'],
    #         'attention_mask': batch_data['attention_mask']
    #     }
    #     labels = batch_data['labels']
    #     return data, labels
    # engine.schedule.data_process_func = data_process_func

    # timier = MultiTimer()

    # trainer = Trainer(engine=engine, logger=logger, timer=timier)

    # hook_list = [
    #     hooks.LossHook(),
    #     hooks.LRSchedulerHook(lr_scheduler=lr_scheduler, by_epoch=True),
    #     hooks.LogMetricByEpochHook(logger),
    #     hooks.ThroughputHook(ignored_steps=10, tflop_per_step=tflop),
    #     hooks.LogMetricByStepHook(),
    #     hooks.LogMemoryByEpochHook(logger),
    # ]

    # trainer.fit(train_dataloader=train_dataloader,
    #             epochs=gpc.config.NUM_EPOCHS,
    #             test_interval=1,
    #             hooks=hook_list,
    #             display_progress=True,
    #             return_output_label=False,
    #             max_steps=5)

if __name__ == '__main__':
    main()