from colossalai.utils import get_dataloader, get_current_device
from colossalai.pipeline.rpc.utils import rpc_run, parse_args
from colossalai.pipeline.rpc import OneFOneBPipelineEngine
from colossalai.pipeline.pipeline_process_group import ppg
from colossalai.pipeline.pipelinable import PipelinableContext
from colossalai.logging import get_dist_logger, disable_existing_loggers
from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai import nn as col_nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoTokenizer, OPTForCausalLM, default_data_collator
from tqdm.auto import tqdm
from itertools import chain
from functools import partial
from colossalai.fx import ColoTracer

import torch
import time

# TODO: Set configs with cmd
# dataset_path = "/data/scratch/huggingface/datasets/wikitext/wikitext-2/"
dataset_path = "wikitext"
dataset_config = "wikitext-2-raw-v1"
model_name = "facebook/opt-125m"
max_block_size = 1024

class OPTLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = CrossEntropyLoss()

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
def get_number_of_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def even_split(lst, n):
    res = []
    size = len(lst) // n
    remain = len(lst) % n
    start = 0
    for i in range(0, n):
        if i < remain:
            res.append(lst[start:start+size+1])
            start += size + 1
        else:
            res.append(lst[start:start+size])
            start += size
    return res

def set_stage_output(mod_graph, nodes, start_node, end_node, next_start, next_end):
    pass

def create_partition_module(pp_rank: int, stage_num: int, model, data_kwargs):
    tracer = ColoTracer()
    meta_args = {k: v.to('meta') for k, v in data_kwargs.items()}
    graph = tracer.trace(root=model, meta_args=meta_args)
    gm = torch.fx.GraphModule(model, graph, model.__class__.__name__)
    mod_graph = gm.graph
    nodes = list(mod_graph.nodes)
    nodes_to_split = []
    for node in nodes:
        next_node = node.next
        if next_node.op == 'call_module' and 'model.decoder.layers' in next_node.target and 'self_attn_layer_norm' in next_node.target:
            nodes_to_split.append(node)
        # if node.op == 'output':
        #     print('---------')
        #     print(f'node: {node} | op: {node.op} | target:{node.target} | args: {node.args} | users: {node.users}')
            # print(f'node: {node.prev} | op: {node.prev.op} | target:{node.prev.target} | args: {node.prev.args} | users: {node.prev.users}')
    
    nodes_in_rank = even_split(nodes_to_split, stage_num)
    start_node = nodes_in_rank[pp_rank][0] if pp_rank > 0 else nodes[0]
    end_node = nodes_in_rank[pp_rank+1][0].prev if pp_rank < stage_num-1 else nodes[-1]
    #print(f'start: {start_node} | end: {end_node}')
    # construct new Module
    # 1. remove tail & add output
    # 2. add input & remove head
    if pp_rank < stage_num - 1:
        for node in reversed(nodes):
            if not end_node is node:
                mod_graph.erase_node(node)
            else:
                break
        with mod_graph.inserting_after(end_node):
            next_start = nodes_in_rank[pp_rank+1][0]
            next_end = nodes_in_rank[pp_rank+2][1].prev if pp_rank < stage_num-2 else nodes[-1]
            set_stage_output(mod_graph, nodes, start_node, end_node, next_start, next_end)

def partition(data_kwargs: dict, pp_rank: int, chunk: int, stage_num: int):
    ppg.initialise_lock.acquire()
    if pp_rank == 0:
        config = AutoConfig.from_pretrained(model_name)
        model = OPTForCausalLM(config)
        
        create_partition_module(pp_rank, stage_num, model, data_kwargs)
        #print(f"rank_{pp_rank} {get_number_of_params(partition) * 4 // 1024 ** 2}M")
    
        ppg.initialise_lock.release()
    exit()
    return partition

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
    logger = get_dist_logger()
    
    batch_size = args.batch_size
    chunk = args.chunk
    device = args.device
    world_size = args.world_size
    stage_num = world_size
    num_microbatches = args.num_microbatches
    epoch = args.epoch
    
    # prepare dataset
    logger.info("Start preparing dataset", ranks=[0])
    raw_datasets = load_dataset(dataset_path, dataset_config)
    logger.info("Dataset is prepared", ranks=[0])
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info(f"{tokenizer.__class__.__name__} has been created", ranks=[0])
    
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
    )
        
    block_size = min(max_block_size, tokenizer.model_max_length)
    
    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i:i + block_size] for i in range(0, total_length, block_size)
                ] for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    train_dataset = lm_datasets["train"]
    # eval_dataset = lm_datasets["validation"]
    
    # DataLoaders creation:
    logger.info("Dataloaders is creating", ranks=[0])
    train_dataloader = get_dataloader(train_dataset,
                                      shuffle=True,
                                      drop_last=True,
                                      pin_memory=True,
                                      collate_fn=default_data_collator,
                                      batch_size=batch_size)
    # eval_dataloader = DataLoader(eval_dataset,
    #                              collate_fn=default_data_collator,
    #                              batch_size=batch_size)
    logger.info("Dataloaders have been created", ranks=[0])
    
    data_kwargs = {},
    for b in train_dataloader:
        data_kwargs = {'input_ids': b['input_ids'],
            'attention_mask': b['attention_mask']}
        break
    
    # engine
    engine = OneFOneBPipelineEngine(
        partition_fn=partial(partition, data_kwargs),
        stage_num=stage_num,
        num_microbatches=num_microbatches,
        device=device,
        chunk=chunk,
        criterion=OPTLoss(),
        metric=None,
        checkpoint=False,
        data_process_func=data_process_func
    )
    
    engine.initialize_optimizer(getattr(torch.optim, args.optimizer), lr=1e-3)
    
    for b in train_dataloader:
        batch = {'input_ids': b['input_ids'],
            'attention_mask': b['attention_mask']}
        labels = b['labels']
        engine.forward_backward(batch=batch, labels=labels)
        break

if __name__ == '__main__':
    disable_existing_loggers()
    args = parse_args()
    rpc_run(args, run_master)