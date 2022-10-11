# Copyright (c) Meta Platforms, Inc. and affiliates
import argparse
import inspect
import logging
import os
import time
from functools import reduce

import torch
from transformers import GPT2LMHeadModel, GPT2Config
from titans.loss.lm_loss import GPTLMLoss
from dataset.webtext import WebtextDataset
import torch.nn as nn
from titans.model.gpt import gpt2_small
from tqdm import tqdm

import pippy.fx
from pippy import run_pippy
from pippy.IR import MultiUseParameterConfig, Pipe, PipeSplitWrapper, annotate_split_points, LossWrapper
from pippy.PipelineDriver import PipelineDriverFillDrain, PipelineDriver1F1B, PipelineDriverInterleaved1F1B, \
    PipelineDriverBase
from pippy.events import EventsContext
from pippy.hf import PiPPyHFTracer
from pippy.microbatch import CustomReducer, TensorChunkSpec
from pippy.visualizer import events_to_json

import colossalai.utils as utils
from colossalai.pipeline.rpc import pytree_map
from colossalai.fx.passes.adding_split_node_pass import uniform_split_pass


logging.disable(logging.INFO)
PROFILING_ENABLED = True
CHECK_NUMERIC_EQUIVALENCE = True


schedules = {
    'FillDrain': PipelineDriverFillDrain,
    '1F1B': PipelineDriver1F1B,
    'Interleaved1F1B': PipelineDriverInterleaved1F1B,
}

VERBOSE = bool(int(os.environ.get('VERBOSE', False)))
logging.shutdown()
if VERBOSE:
    logging.getLogger().setLevel(logging.DEBUG)

pippy.fx.Tracer.proxy_buffer_attributes = True


def get_number_of_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# TODO: Fails! Why??? https://gist.github.com/pbelevich/f4e78c6ed2fdabc8b02ab15e254935fd
# def add_split_points(gpt2, layers_per_rank):
#     for i in range(0, gpt2.config.n_layer // layers_per_rank):
#         annotate_split_points(gpt2, {f'transformer.h.{i * layers_per_rank}': PipeSplitWrapper.SplitPoint.BEGINNING})
#     annotate_split_points(gpt2, {
#         f'transformer.h.{gpt2.config.n_layer // layers_per_rank - 1}': PipeSplitWrapper.SplitPoint.END})
#     return gpt2.config.n_layer // layers_per_rank + 2



def add_split_points(gpt2, decoders_per_rank):
    for i in range(0, gpt2.config.n_layer // decoders_per_rank):
        annotate_split_points(gpt2, {f'transformer.h.{i * decoders_per_rank}': PipeSplitWrapper.SplitPoint.BEGINNING})
    annotate_split_points(gpt2, {f'transformer.ln_f': PipeSplitWrapper.SplitPoint.BEGINNING})
    return gpt2.config.n_layer // decoders_per_rank + 2


class OutputLossWrapper(LossWrapper):
    def __init__(self, module, loss_fn):
        super().__init__(module, loss_fn)
        for attr in module.__dir__():
            if not hasattr(self, attr):
                setattr(self, attr, getattr(module, attr))

    def forward(self, input, target):
        output = self.module(input)
        return output, self.loss_fn(output, target)

def run_master(_, args):

    chunk = args.chunk
    epoch = args.epoch
    batch_size = args.batch_size
    
    MULTI_USE_PARAM_CONFIG = MultiUseParameterConfig.REPLICATE if args.replicate else MultiUseParameterConfig.TRANSMIT
    print(f'REPLICATE config: {args.replicate} -> {MULTI_USE_PARAM_CONFIG}')
    print("Using schedule:", args.schedule)

    assert args.world_size >= 4, "This program requires at least 3 workers + 1 master"

    # see titans.model.gpt.GPT
    gpt2 = GPT2LMHeadModel(GPT2Config(vocab_size=50304))
    # gpt2 = OutputLossWrapper(gpt2, loss_fn=GPTLMLoss)

    print(f"GPT-2 total number of params = {get_number_of_params(gpt2) * 4 // 1024 ** 2}M")


    emb_head = 2  # embeddings + head
    master_emb_head = 1 + emb_head  # master + embeddings + head
    decoders_per_rank = (gpt2.config.n_layer + (args.world_size - master_emb_head) - 1) // (
            args.world_size - master_emb_head)  # a divider of gpt2.config.n_layer: [1, 2, 3, 4, 6, 12]

    number_of_workers = emb_head + gpt2.config.n_layer // decoders_per_rank  # 3 + a divider of gpt2.config.n_layer: [4, 5, 6, 7, 9, 15]

    all_worker_ranks = list(range(1, 1 + number_of_workers))  # exclude master rank = 0



    device = args.device

    train_ds = WebtextDataset(os.environ['DATA'], seq_len=1024)
    train_dataloader = utils.get_dataloader(train_ds,
                                            seed=42,
                                            batch_size=batch_size,
                                            pin_memory=True,
                                            shuffle=True,
                                            drop_last=True)


    sm_cnt = add_split_points(gpt2, decoders_per_rank)
    assert sm_cnt == len(all_worker_ranks), f"sm_cnt = {sm_cnt} all_worker_ranks = {all_worker_ranks}"

    # print(gpt2)

    input_names = ['input_ids', 'labels', 'attention_mask']
    sig = inspect.signature(gpt2.forward)
    concrete_args = {p.name: p.default for p in sig.parameters.values() if p.name not in input_names}

    print('Instantiating GPT-2 Pipeline')
    output_loss_value_spec = {'loss': True, 'logits': False,
                              'past_key_values': [[False for _ in range(2)] for _ in range(12)]}


    gpt2_pipe = Pipe.from_tracing(gpt2, MULTI_USE_PARAM_CONFIG, tracer=PiPPyHFTracer(), concrete_args=concrete_args,
                                  output_loss_value_spec=output_loss_value_spec, deep_copy_module=False)
    assert sm_cnt == len(list(gpt2_pipe.split_gm.children()))
    gpt2_pipe.to(device)

    # for sm in gpt2_pipe.split_gm.children():
    #     print(dict(sm.named_modules()).keys())
    #     print('-' * 20)

    # return

    # kernel choose
    for bx, by in train_dataloader:
        current_batch = {
            "input_ids" : bx["input_ids"].to(device),
            "labels" : by.to(device),
            "attention_mask" : bx["attention_mask"].to(device)
        }
        break

    gpt2_pipe(**current_batch)

    for i, sm in enumerate(gpt2_pipe.split_gm.children()):
        print(f"submod_{i} {get_number_of_params(sm) * 4 // 1024 ** 2}M params")


    args_chunk_spec = ()
    kwargs_chunk_spec = {'input_ids': TensorChunkSpec(0), 'labels': TensorChunkSpec(0), 'attention_mask': TensorChunkSpec(0)}
    output_chunk_spec = {'loss': CustomReducer(torch.tensor(0.0), lambda a, b: a + b), 'logits': TensorChunkSpec(0),
                         'past_key_values': [[TensorChunkSpec(0) for _ in range(2)] for _ in range(12)]}
    pipe_driver: PipelineDriverBase = schedules[args.schedule](gpt2_pipe, chunk, args_chunk_spec, kwargs_chunk_spec,
                                                               output_chunk_spec, len(all_worker_ranks),
                                                               all_ranks=all_worker_ranks,
                                                               _debug_mask_minibatches=False,
                                                               _record_mem_dumps=bool(args.record_mem_dumps),
                                                               checkpoint=bool(args.checkpoint))

    this_file_name = os.path.splitext(os.path.basename(__file__))[0]

    print('Running GPT2 pipeline.')
    pipe_visualized_filename = f"{this_file_name}_visualized.json"
    

    for epoch_id in range(epoch):
        times = []

        for bx, by in tqdm(train_dataloader):
            s = time.time()
            current_batch = {
                "input_ids" : bx["input_ids"].to(device),
                "labels" : by.to(device),
                "attention_mask" : bx["attention_mask"].to(device)
            }
            pipe_driver(**current_batch)
            cost_time = time.time() - s
            times.append(cost_time)

            if len(times) == 10:
                break
        
        print("avg cost time : {}s".format(sum(times) / len(times)))
        break
            



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=int(os.getenv("WORLD_SIZE", 16)))
    parser.add_argument('--rank', type=int, default=int(os.getenv("RANK", -1)))
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--dp_degree', type=int, default=1)
    parser.add_argument('--tp_degree', type=int, default=1)
    parser.add_argument('--num_microbatches', type=int, default=2)
    
    parser.add_argument('--chunk', type=int, default=1)
    parser.add_argument('--master_addr', type=str, default=os.getenv('MASTER_ADDR', 'localhost'))
    parser.add_argument('--master_port', type=str, default=os.getenv('MASTER_PORT', '29500'))
    parser.add_argument('-s', '--schedule', type=str, default=list(schedules.keys())[0], choices=schedules.keys())
    parser.add_argument('--replicate', type=int, default=int(os.getenv("REPLICATE", '0')))
    parser.add_argument('--cuda', type=int, default=int(torch.cuda.is_available()))
    parser.add_argument('--record_mem_dumps', type=int, default=0, choices=[0, 1])
    parser.add_argument('--checkpoint', type=int, default=0, choices=[0, 1])
    args = parser.parse_args()

    run_pippy(run_master, args)
