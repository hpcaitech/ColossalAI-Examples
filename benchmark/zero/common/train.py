import math
import time

import torch
from torch.distributed import all_reduce, get_rank, get_world_size
from tqdm import tqdm

from zero.common.utils import CONFIG, AsyncMemoryMonitor, print_log, get_tflops


def _train(epoch, rank, world_size, train_dataloader, model, criterion, optimizer, lr_scheduler, scaler, mem_monitor):
    use_optimizer_backward = CONFIG['method'] in ['colossalai']
    use_integrated_backward = CONFIG['method'] in ['deepspeed', 'patrickstar']
    use_integrated_step = CONFIG['method'] in ['deepspeed']
    use_autocast = CONFIG['method'] in ['torch', 'colossalai'] and \
        'fp16' in CONFIG and CONFIG['fp16'].get('enabled', True)
    clip_grad_norm = CONFIG.get('gradient_clipping', 0.)
    use_integrated_clip_grad = CONFIG['method'] in ['fairscale']
    use_colossalai_zero_v1 = CONFIG['method'] == 'colossalai' and CONFIG.get('sharded_model_version', 2) == 1

    model.train()

    num_steps = len(train_dataloader)
    if 'steps_per_epoch' in CONFIG['hyperparameter'] and CONFIG['hyperparameter']['steps_per_epoch'] < num_steps:
        num_steps = CONFIG['hyperparameter']['steps_per_epoch']
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

        if use_colossalai_zero_v1:
            model.zero_grad(set_to_none=True)

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

        if use_autocast:
            with torch.cuda.amp.autocast():
                outputs = model(**batch)
        else:
            outputs = model(**batch)

        loss = criterion(outputs, labels)
        train_loss += loss

        fwd_end = time.time()

        bwd_start = time.time()

        if use_colossalai_zero_v1:
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
        elif use_integrated_backward:    # deepspeed & patrickstar style
            model.backward(loss)
            if use_integrated_step:
                model.step()    # deepspeed style
            else:
                optimizer.step()    # patrickstar style
                lr_scheduler.step()

        elif use_optimizer_backward:    # colossalai style
            optimizer.backward(loss)
            if clip_grad_norm > 0:
                optimizer.clip_grad_norm(model, clip_grad_norm)
            optimizer.step()
            lr_scheduler.step()

        elif scaler is not None:    # torch & fairscale amp style
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if clip_grad_norm > 0:
                if use_integrated_clip_grad:    # fairscale style
                    model.clip_grad_norm_(clip_grad_norm)
                else:    # torch style
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

        else:    # torch & fairscale normal style
            loss.backward()
            if clip_grad_norm > 0:
                if use_integrated_clip_grad:    # fairscale style
                    model.clip_grad_norm_(clip_grad_norm)
                else:    # torch style
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
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
                                 tflops=get_tflops(batch_time, batch_tokens * world_size))

    peak_mem = None
    if mem_monitor is not None:
        peak_mem = max(mem_monitor.finish())

    all_reduce(train_loss)
    all_reduce(num_samples)
    all_reduce(num_tokens)

    msg = f'[Epoch {epoch} / Train]: Loss = {train_loss.item() / (world_size * num_steps):.3f}'
    msg += f' | Throughput = {num_samples.item() / (used_time + 1e-12):.3f} samples/sec'
    msg += f' | TFLOPS = {get_tflops(used_time, num_tokens.item()):.3f}'
    if peak_mem is not None:
        msg += f' | Peak memory = {peak_mem / 1024:.3f} GB.'
    print_log(msg)


def _test(epoch, rank, world_size, test_dataloader, model, criterion, mem_monitor):
    use_autocast = CONFIG['method'] in ['torch', 'colossalai'] and \
        'fp16' in CONFIG and CONFIG['fp16'].get('enabled', True)
    evaluation = CONFIG['model']['evaluation']

    model.eval()

    num_steps = len(test_dataloader)
    if 'steps_per_epoch' in CONFIG['hyperparameter'] and CONFIG['hyperparameter']['steps_per_epoch'] < num_steps:
        num_steps = CONFIG['hyperparameter']['steps_per_epoch']
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
            if use_autocast:
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
            else:
                outputs = model(**batch)

            loss = criterion(outputs, labels)
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
                               tflops=get_tflops(batch_time, batch_tokens * world_size))
                if evaluation == 'ppl':
                    metrics['perplexity'] = math.exp(loss.item())
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
    msg += f' | TFLOPS = {get_tflops(used_time, num_tokens.item()):.3f}'
    if peak_mem is not None:
        msg += f' | Peak memory = {peak_mem / 1024:.3f} GB.'
    print_log(msg)


def train(model, train_data, test_data, criterion, optimizer, scaler, lr_scheduler):
    rank = get_rank()
    world_size = get_world_size()

    mem_monitor = None
    if CONFIG.get('use_mem_monitor'):
        mem_monitor = AsyncMemoryMonitor(rank)

    numel = CONFIG['model']['numel']
    if numel < 1e9:
        msg = f'{numel / 1e6:.3f} M'
    else:
        msg = f'{numel / 1e9:.3f} B'
    print_log(f'Model is built (parameter size = {msg}).')

    print_log('Benchmark start.')

    for epoch in range(CONFIG['hyperparameter']['num_epochs']):
        _train(epoch, rank, world_size, train_data, model, criterion, optimizer, lr_scheduler, scaler, mem_monitor)
        _test(epoch, rank, world_size, test_data, model, criterion, mem_monitor)

    print_log('Benchmark complete.')
