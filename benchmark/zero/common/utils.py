import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from torch.distributed import get_rank, is_initialized

CONFIG = dict()


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args = parser.parse_args()

    config_file = args.config

    assert os.path.exists(config_file), 'No valid config file found.'

    with open(config_file, 'r') as f:
        cfg = json.load(f)
        for k, v in cfg.items():
            CONFIG[k] = v


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
        self.file = os.path.join(CONFIG['log_path'], f'memory_rank_{rank}.log') if save_to_disk else None

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
            gpu_usage.append(torch.cuda.max_memory_allocated(self.rank) / (1024 * 1024))    # MB
            torch.cuda.reset_peak_memory_stats(self.rank)
            time.sleep(self.interval)

        return gpu_usage


def print_log(msg):
    msg = f'{time.asctime()} > {msg}'
    rank = get_rank() if is_initialized() else 0
    log_file = os.path.join(CONFIG['log_path'], f'training_rank_{rank}.log')
    with open(log_file, 'a') as f:
        f.write(msg + '\n')
    if rank == 0:
        print(msg)


class ModelFromHF(torch.nn.Module):

    def __init__(self, config, model_cls):
        super().__init__()
        self.module = model_cls(config)
        if CONFIG['model'].get('checkpoint'):
            self.module.apply(self.set_checkpointing)

    def set_checkpointing(self, module):
        if hasattr(module, 'gradient_checkpointing'):
            module.gradient_checkpointing = True

    def forward(self, *args, **kwargs):
        output = self.module(*args, **kwargs)
        return output.logits


def get_tflops(iter_time: float, num_tokens: int) -> float:
    flops = CONFIG['model']['numel'] * num_tokens * 2.0 * 4.0
    return (flops / 1e12) / (iter_time + 1e-12)


def get_model_size(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters())


def get_gpu_memory_mb():
    return torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 1024**2
