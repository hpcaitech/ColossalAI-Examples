from typing import List
import time
import os
import inspect
import logging

import torch
from titans.loss.lm_loss import GPTLMLoss
from dataset.webtext import WebtextDataset
import torch.nn as nn
from titans.model.gpt import gpt2_small
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch.autograd.profiler as profiler


from colossalai.pipeline.rpc.utils import rpc_run, parse_args
from colossalai.pipeline.rpc import OneFOneBPipelineEngine, pytree_map
from colossalai import nn as col_nn
from colossalai.pipeline.pipelinable import PipelinableContext, PipelinableModel
from colossalai.pipeline.pipeline_process_group import ppg
import colossalai.utils as utils


torch.manual_seed(1024)
logging.disable(logging.INFO)

def get_number_of_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def mask_function(attention_mask=None):
    batch_size = ppg.args.batch_size
    num_microbatches = ppg.args.num_microbatches
    # print(pytree_map(attention_mask, lambda x : x.shape, process_types=torch.Tensor))
    if attention_mask is not None:
        microbatch_size = batch_size // num_microbatches
        attention_mask = attention_mask.view(microbatch_size, -1)
        attention_mask = col_nn.partition_batch(attention_mask)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = (1.0 - attention_mask) * -10000.0
    return attention_mask


def partition(pp_rank: int, chunk: int, stage_num: int):
    ppg.initialise_lock.acquire()
    pipelinable = PipelinableContext(policy='uniform')
    with pipelinable:
        model = gpt2_small()

    exec_seq = ['embed', mask_function, 'blocks.0', 'blocks.1', mask_function, 'blocks.2', 'blocks.3', 'blocks.4', 'blocks.5', (mask_function, "front"), \
            'blocks.6', 'blocks.7', 'blocks.8', 'blocks.9', 'blocks.10', 'blocks.11', 'norm', 'head']
    pipelinable.to_layer_list(exec_seq)
    
    partition = pipelinable.partition(1, stage_num, pp_rank)
    print(f"rank_{pp_rank} {get_number_of_params(partition) * 4 // 1024 ** 2}M")
    
    ppg.initialise_lock.release()
    
    return partition


def calc_local_model_size(model: torch.nn.Module):
    numel_per_device = 0
    for p in model.parameters():
        numel_per_device += p.numel()
    return numel_per_device

# world_size = 4 for uniform
def data_process_func(pp_rank: int, args_kwargs):
    if pp_rank == 0:
        args = args_kwargs[0]
        kwargs = args_kwargs[1]
        return args, kwargs

    elif pp_rank == 1:
        x = args_kwargs[0]
        attention_mask = args_kwargs[1]
        args = [x]
        kwargs = {"attention_mask" : attention_mask}
        return args, kwargs
    
    elif pp_rank == 2:
        x = args_kwargs[0]
        attention_mask = args_kwargs[1]
        args = [x]
        kwargs = {"attention_mask" : attention_mask}
        return args, kwargs

    elif pp_rank == 3:
        x = args_kwargs[0]
        attention_mask = args_kwargs[1]
        args = [x]
        kwargs = {"attention_mask" : attention_mask}
        return args, kwargs


def run_master(args):
    batch_size = args.batch_size
    chunk = args.chunk
    device = args.device
    world_size = args.world_size
    stage_num = world_size
    num_microbatches = args.num_microbatches
    epoch = args.epoch


    train_ds = WebtextDataset(os.environ['DATA'], seq_len=1024)
    train_dataloader = utils.get_dataloader(train_ds,
                                            seed=42,
                                            batch_size=batch_size,
                                            pin_memory=True,
                                            shuffle=True,
                                            drop_last=True)


    criterion = GPTLMLoss()

    engine = OneFOneBPipelineEngine(
        partition_fn=partition,
        stage_num=stage_num,
        num_microbatches=num_microbatches,
        device=device,
        chunk=chunk,
        criterion=criterion,
        metric=None,
        checkpoint=False,
        data_process_func=data_process_func
    )

    engine.initialize_optimizer(getattr(torch.optim, args.optimizer), lr=1e-3)


    # choose kernel
    for bx, by in train_dataloader:
        engine.forward_backward(bx, by)
        break
    
    # with profiler.profile(enabled=True, use_cpu=True, use_cuda=True, use_kineto=False) as p:
    
    for epoch_id in range(epoch):
        data_iter = tqdm(train_dataloader, desc=f'[Train->Epoch {epoch_id}]')

        times = []
        for bx, by in data_iter:
            s = time.time()
            engine.forward_backward(bx, by)
            cost_time = time.time() - s
            times.append(cost_time)

            if len(times) == 10:
                break
            # batch_info = dict()
            # avg_loss = sum(losses) / len(losses)
            # batch_info['avg_loss'] = avg_loss
            # if metrics is not None and metrics[0] is not None:
            #     avg_metric = sum(metrics) / len(metrics)
            #     batch_info['avg_metric'] = avg_metric
            # data_iter.set_postfix(batch_info)

        print("avg cost time : {}s".format(sum(times) / len(times)))
        break
    
    # print(p.key_averages().table())
    # p.export_chrome_trace("trace.json")


if __name__ == '__main__':
    args = parse_args()
    rpc_run(args, run_master)
