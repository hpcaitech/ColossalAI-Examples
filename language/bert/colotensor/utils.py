from tqdm import tqdm
import time
import math
import os

import colossalai.utils as utils
import torch
from torch.distributed import all_reduce, get_rank, get_world_size, is_initialized
from colossalai.utils import colo_set_process_memory_fraction, get_current_device, MultiTimer
from colossalai.core import global_context as gpc
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

def train(epoch, rank, world_size, train_dataloader, model, criterion, optimizer, lr_scheduler, mem_monitor, numel, use_zero):
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

        if use_zero:
            loss = loss.float()
            optimizer.backward(loss)
        else:
            loss.backward()
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

def test(epoch, rank, world_size, test_dataloader, model, criterion, mem_monitor, numel):
    evaluation = 'ppl'

    model.eval()

    num_steps = len(test_dataloader)
    progress = range(num_steps)
    if rank == 0:
        progress = tqdm(progress, desc=f"[Epoch {epoch} / Test]")

    test_loss = torch.zeros(()).to(torch.float).to(rank)
    used_time = 0.
    num_steps = 0
    num_samples = torch.zeros(()).to(torch.int).to(rank)
    num_tokens = torch.zeros(()).to(torch.int).to(rank)
    correct = torch.zeros(()).to(torch.int).to(rank)

    data_iter = iter(test_dataloader)

    if mem_monitor is not None:
        mem_monitor.start()

    with torch.no_grad():
        for _ in progress:
            batch_start = time.time()

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
            test_loss += loss

            batch_end = time.time()

            num_steps += 1
            num_samples += batch_size
            num_tokens += batch_tokens

            batch_time = batch_end - batch_start
            used_time += batch_time

            if rank == 0:
                metrics = dict(loss=loss.item(),
                               step_time=batch_time,
                               throughput=batch_size * world_size / (batch_time + 1e-12),
                               tflops=get_tflops(batch_time, batch_tokens * world_size, numel))
                if evaluation == 'ppl':
                    metrics['loss'] = loss.item()
                elif evaluation == 'acc':
                    if not isinstance(labels, torch.Tensor):
                        labels = labels['targets_a']
                    batch_correct = torch.sum(labels == torch.argmax(outputs, dim=-1)).item()
                    metrics['accuracy'] = batch_correct / batch_size
                    correct += batch_correct
                else:
                    raise ValueError(f'Invalid evaluation method {evaluation}')
                progress.set_postfix(**metrics)

    peak_mem = None
    if mem_monitor is not None:
        peak_mem = max(mem_monitor.finish())

    all_reduce(test_loss)
    reduced_loss = test_loss.item() / (world_size * num_steps)
    all_reduce(num_samples)
    all_reduce(num_tokens)
    if evaluation == 'acc':
        all_reduce(correct)

    msg = f'[Epoch {epoch} / Test]: Loss = {reduced_loss:.3f}'
    if evaluation == 'ppl':
        msg += f' | Perplexity = {math.exp(reduced_loss):.3f}'
    else:
        msg += f' | Accuracy = {correct.item() * 100 / num_samples.item():.3f} %'
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

