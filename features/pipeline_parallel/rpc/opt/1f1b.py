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
from colossalai.fx.passes.adding_split_node_pass import split_with_split_nodes_pass, balanced_split_pass, avgnode_split_pass
from colossalai.pipeline.middleware.adaptor import get_fx_topology

import torch
import time
import inspect
'''
Add following code to colossalai/pipeline/rpc/_pipeline_base.py +497 to make the code work
if not grad_tensors is None:
    real_out_len = len(grad_tensors)
    if not isinstance(stage_outputs, torch.Tensor) and  len(grad_tensors) < len(stage_outputs):
        stage_outputs = stage_outputs[:real_out_len]
'''

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
        logits = logits[0]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


def create_partition_module(pp_rank: int, stage_num: int, model, data_kwargs):
    tracer = ColoTracer()
    meta_args = {k: v.to('meta') for k, v in data_kwargs.items()}
    graph = tracer.trace(root=model, meta_args=meta_args)
    gm = torch.fx.GraphModule(model, graph, model.__class__.__name__)
    annotated_model = avgnode_split_pass(gm, stage_num)

    top_module, split_submodules = split_with_split_nodes_pass(annotated_model, merge_output=True)
    topo = get_fx_topology(top_module)
    for submodule in split_submodules:
        if isinstance(submodule, torch.fx.GraphModule):
            setattr(submodule, '_topo', topo)
    return split_submodules[pp_rank + 1]


def partition(data_kwargs: dict, pp_rank: int, chunk: int, stage_num: int):
    config = AutoConfig.from_pretrained(model_name, local_files_only=True)
    config.enable_bias = False    # temporarily set it for tracer bug
    model = OPTForCausalLM(config)
    module = create_partition_module(pp_rank, stage_num, model, data_kwargs)
    num_params = sum(param.numel() for param in module.parameters())
    print(f'{pp_rank=} {num_params}')
    return module


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

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
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
        data_kwargs = {'input_ids': b['input_ids'], 'attention_mask': b['attention_mask']}
        break

    # engine
    engine = OneFOneBPipelineEngine(
        partition_fn=partial(partition, data_kwargs),
        stage_num=stage_num,
        num_microbatches=num_microbatches,
        device=device,
        chunk=chunk,
        criterion=OPTLoss(),
        checkpoint=False,
    )

    engine.initialize_optimizer(getattr(torch.optim, args.optimizer), lr=1e-3)

    fwd_cnt = 0
    for b in train_dataloader:
        batch = {'input_ids': b['input_ids'], 'attention_mask': b['attention_mask']}
        labels = b['labels']
        res = engine.forward_backward(batch=batch, labels=labels, forward_only=False)
        fwd_cnt += 1
        print(f'iter: {fwd_cnt} | {res=}')


if __name__ == '__main__':
    disable_existing_loggers()
    args = parse_args()
    rpc_run(args, run_master)
