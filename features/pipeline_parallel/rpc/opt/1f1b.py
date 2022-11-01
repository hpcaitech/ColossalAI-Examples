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
    config = AutoConfig.from_pretrained(model_name)
    
    ppg.initialise_lock.acquire()
    pipelinable = PipelinableContext(policy='uniform')
    with pipelinable:
        model = OPTForCausalLM.from_pretrained(
                model_name,
                from_tf=bool(".ckpt" in model_name),
                config=config,
                local_files_only=False
            )

    #exec_seq = ['embed', mask_function, 'blocks.0', 'blocks.1', mask_function, 'blocks.2', 'blocks.3', 'blocks.4', 'blocks.5', (mask_function, "front"), \
    #        'blocks.6', 'blocks.7', 'blocks.8', 'blocks.9', 'blocks.10', 'blocks.11', 'norm', 'head']
    exec_seq = ['model.decoder', 'lm_head']
    pipelinable.to_layer_list(exec_seq)
    
    partition = pipelinable.partition(1, stage_num, pp_rank)
    print(f"rank_{pp_rank} {get_number_of_params(partition) * 4 // 1024 ** 2}M")
    
    ppg.initialise_lock.release()
    
    return partition


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
    eval_dataset = lm_datasets["validation"]
    
    # DataLoaders creation:
    logger.info("Dataloaders is creating", ranks=[0])
    train_dataloader = get_dataloader(train_dataset,
                                      shuffle=True,
                                      drop_last=True,
                                      pin_memory=True,
                                      collate_fn=default_data_collator,
                                      batch_size=batch_size)
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=default_data_collator,
                                 batch_size=batch_size)
    logger.info("Dataloaders have been created", ranks=[0])
    
    # engine
    engine = OneFOneBPipelineEngine(
        partition_fn=partition,
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
    
    # choose kernel
    for b in train_dataloader:
        batch = {'input_ids': b['input_ids'],
            'attention_mask': b['attention_mask']}
        labels = b['labels']
        engine.forward_backward(batch=batch, labels=labels)
        break
    
    for epoch_id in range(epoch):
        data_iter = tqdm(train_dataloader, desc=f'[Train->Epoch {epoch_id}]')

        times = []
        for b in data_iter:
            batch = {'input_ids': b['input_ids'],
                'attention_mask': b['attention_mask']}
            labels = b['labels']
            s = time.time()
            engine.forward_backward(batch=batch, labels=labels)
            cost_time = time.time() - s
            times.append(cost_time)

            if len(times) == 10:
                break

        print("avg cost time : {}s".format(sum(times) / len(times)))
        break

if __name__ == '__main__':
    disable_existing_loggers()
    args = parse_args()
    rpc_run(args, run_master)