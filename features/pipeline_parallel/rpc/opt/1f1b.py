from colossalai.utils import get_dataloader, get_current_device
from colossalai.pipeline.rpc.utils import rpc_run, parse_args
from colossalai.logging import get_dist_logger, disable_existing_loggers
from colossalai.utils.model.colo_init_context import ColoInitContext
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, OPTForCausalLM, default_data_collator
from itertools import chain

# TODO: Set configs with cmd
# dataset_path = "/data/scratch/huggingface/datasets/wikitext/wikitext-2/"
dataset_path = "wikitext"
dataset_config = "wikitext-2-raw-v1"
model_name = "facebook/opt-125m"
max_block_size = 1024

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
    
    config = AutoConfig.from_pretrained(model_name)
    logger.info("Model config has been created", ranks=[0])
    
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
                                      add_sampler=True,
                                      drop_last=True,
                                      pin_memory=True,
                                      collate_fn=default_data_collator,
                                      batch_size=batch_size)
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=default_data_collator,
                                 batch_size=batch_size)
    logger.info("Dataloaders have been created", ranks=[0])

if __name__ == '__main__':
    disable_existing_loggers()
    args = parse_args()
    rpc_run(args, run_master)