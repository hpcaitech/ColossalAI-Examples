import random
import torch
import numpy as np
import copy

from itertools import chain
from datasets import load_from_disk, set_progress_bar_enabled

from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed import get_world_size

from transformers import BertTokenizer, DataCollatorForLanguageModeling
from colossalai.logging import get_dist_logger


def build_data_from_wikitext(dataset_path: str, tokenizer_path: str, seq_len: int = 512, batch_size: int = 8):
    logger = get_dist_logger("build_data_from_wikitext")
    logger.info("Building Wikitext-2 ...", ranks=[0])
    world_size = get_world_size()

    set_progress_bar_enabled(False)
    dataset = load_from_disk(dataset_path)

    tokenizer = BertTokenizer(vocab_file=tokenizer_path + "/vocab.txt")

    def tokenize(examples):
        seq_length = seq_len
        examples = tokenizer(examples["text"])
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= seq_length:
            total_length = (total_length // seq_length) * seq_length

        result = {
            k: [t[i : i + seq_len] for i in range(0, total_length, seq_length)]
            for k, t in concatenated_examples.items()
        }

        return result

    tokenized_dataset = dataset.map(
        tokenize, batched=True, num_proc=16, load_from_cache_file=False, keep_in_memory=True, remove_columns="text"
    )

    def seed_worker():
        worker_seed = 1024
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        random.seed(worker_seed)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    train_sampler = DistributedSampler(tokenized_dataset['train'], shuffle=True) if world_size > 1 else None
    train_data = DataLoader(tokenized_dataset['train'],
                            shuffle=(train_sampler is None),
                            sampler=train_sampler,
                            drop_last=True,
                            collate_fn=data_collator,
                            worker_init_fn=seed_worker,
                            batch_size=batch_size,
                            pin_memory=True)
    test_sampler = DistributedSampler(tokenized_dataset['validation'], shuffle=False) if world_size > 1 else None
    test_data = DataLoader(tokenized_dataset['validation'],
                           sampler=test_sampler,
                           drop_last=True,
                           collate_fn=data_collator,
                           worker_init_fn=seed_worker,
                           batch_size=batch_size,
                           pin_memory=True)

    return train_data, test_data