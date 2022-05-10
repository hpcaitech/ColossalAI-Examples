import os

import torch
from torch.distributed import get_world_size
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

from zero.common.utils import CONFIG, ModelFromHF, get_model_size

_gpt2_small = dict(
    seq_length=1024,
    vocab_size=50257,
    hidden_size=768,
    num_heads=12,
    depth=12,
    numel=124439808,
    checkpoint=True,
    evaluation='ppl',
)

_gpt2_xl = dict(
    seq_length=1024,
    vocab_size=50257,
    hidden_size=1600,
    num_heads=25,
    depth=48,
    numel=1557611200,
    checkpoint=True,
    evaluation='ppl',
)

_gpt2_10b = dict(
    seq_length=1024,
    vocab_size=50257,
    hidden_size=4096,
    num_heads=16,
    depth=50,
    numel=10279047168,
    checkpoint=True,
    evaluation='ppl',
)

_gpt2_4b = dict(
    seq_length=1024,
    vocab_size=50257,
    hidden_size=2304,
    num_heads=16,
    depth=64,
    checkpoint=True,
    evaluation='ppl',
)

_gpt2_2b = dict(
    seq_length=1024,
    vocab_size=50257,
    hidden_size=2048,
    num_heads=16,
    depth=40,
    checkpoint=True,
    evaluation='ppl',
)

_gpt2_3b = dict(
    seq_length=1024,
    vocab_size=50257,
    hidden_size=2560,
    num_heads=40,
    depth=24,
    checkpoint=True,
    evaluation='ppl',
)

_gpt2_6b = dict(
    seq_length=1024,
    vocab_size=50257,
    hidden_size=4096,
    num_heads=16,
    depth=30,
    checkpoint=True,
    evaluation='ppl',
)

_gpt2_8b = dict(
    seq_length=1024,
    vocab_size=50257,
    hidden_size=4096,
    num_heads=16,
    depth=40,
    checkpoint=True,
    evaluation='ppl',
)

_gpt2_12b = dict(
    seq_length=1024,
    vocab_size=50257,
    hidden_size=4096,
    num_heads=16,
    depth=60,
    checkpoint=True,
    evaluation='ppl',
)

_gpt2_15b = dict(
    seq_length=1024,
    vocab_size=50257,
    hidden_size=4096,
    num_heads=16,
    depth=78,
    checkpoint=True,
    evaluation='ppl',
)

_gpt2_18b = dict(
    seq_length=1024,
    vocab_size=50257,
    hidden_size=4096,
    num_heads=16,
    depth=90,
    checkpoint=True,
    evaluation='ppl',
)

_gpt2_20b = dict(
    seq_length=1024,
    vocab_size=50257,
    hidden_size=8192,
    num_heads=16,
    depth=25,
    checkpoint=True,
    evaluation='ppl',
)

_gpt2_24b = dict(
    seq_length=1024,
    vocab_size=50257,
    hidden_size=8192,
    num_heads=16,
    depth=30,
    checkpoint=True,
    evaluation='ppl',
)

_gpt2_28b = dict(
    seq_length=1024,
    vocab_size=50257,
    hidden_size=8192,
    num_heads=16,
    depth=35,
    checkpoint=True,
    evaluation='ppl',
)

_gpt2_32b = dict(
    seq_length=1024,
    vocab_size=50257,
    hidden_size=8192,
    num_heads=16,
    depth=40,
    checkpoint=True,
    evaluation='ppl',
)

_gpt2_36b = dict(
    seq_length=1024,
    vocab_size=50257,
    hidden_size=8192,
    num_heads=16,
    depth=45,
    checkpoint=True,
    evaluation='ppl',
)

_gpt2_40b = dict(
    seq_length=1024,
    vocab_size=50257,
    hidden_size=8192,
    num_heads=16,
    depth=50,
    checkpoint=True,
    evaluation='ppl',
)

_gpt2_configurations = dict(gpt2=_gpt2_small,
                            gpt2_small=_gpt2_small,
                            gpt2_xl=_gpt2_xl,
                            gpt2_10b=_gpt2_10b,
                            gpt2_4b=_gpt2_4b,
                            gpt2_6b=_gpt2_6b,
                            gpt2_8b=_gpt2_8b,
                            gpt2_2b=_gpt2_2b,
                            gpt2_3b=_gpt2_3b,
                            gpt2_12b=_gpt2_12b,
                            gpt2_15b=_gpt2_15b,
                            gpt2_18b=_gpt2_18b,
                            gpt2_20b=_gpt2_20b,
                            gpt2_40b=_gpt2_40b,
                            gpt2_24b=_gpt2_24b,
                            gpt2_28b=_gpt2_28b,
                            gpt2_32b=_gpt2_32b,
                            gpt2_36b=_gpt2_36b)

_default_hyperparameters = dict(
    tokenize_mode='concat',
    batch_size=4,
    learning_rate=0.00015,
    weight_decay=1e-2,
    num_epochs=2,
    warmup_epochs=1,
    steps_per_epoch=100,
)


def build_data():
    import copy
    import random
    from functools import partial
    from itertools import chain

    import numpy as np
    from datasets import load_from_disk, set_progress_bar_enabled
    from torch.utils.data import DataLoader, Dataset, DistributedSampler
    from transformers import default_data_collator

    world_size = get_world_size()

    if CONFIG['hyperparameter'].get('synthetic'):

        class SyntheticDataset(Dataset):

            def __init__(self, vocab_size, seq_length, size) -> None:
                super().__init__()
                self.size = size
                self.vocab_size = vocab_size
                self.seq_length = seq_length

            def __len__(self):
                return self.size

            def __getitem__(self, _):
                input_ids = torch.randint(self.vocab_size, (self.seq_length,))
                attention_mask = torch.ones((self.seq_length,), dtype=torch.long)
                return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': input_ids.clone()}

        vocab_size = CONFIG['model']['vocab_size']
        seq_len = CONFIG['model']['seq_length']
        train_size = CONFIG['hyperparameter']['batch_size'] * world_size * CONFIG['hyperparameter'].get(
            'steps_per_epoch', 100)
        test_size = CONFIG['hyperparameter']['batch_size'] * world_size * CONFIG['hyperparameter'].get(
            'steps_per_epoch', 10)
        tokenized_dataset = {
            'train': SyntheticDataset(vocab_size, seq_len, train_size),
            'validation': SyntheticDataset(vocab_size, seq_len, test_size)
        }

    else:
        set_progress_bar_enabled(False)
        dataset = load_from_disk(CONFIG['dataset'])
        tokenizer = GPT2Tokenizer(vocab_file=CONFIG['tokenizer'] + '/vocab.json',
                                  merges_file=CONFIG['tokenizer'] + '/merges.txt')

        def tokenize(examples, mode='concat'):
            assert mode in ['concat', 'pad']
            seq_len = CONFIG['model']['seq_length']
            if mode == 'concat':
                examples = tokenizer(examples['text'])
                concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
                total_length = len(concatenated_examples[list(examples.keys())[0]])
                if total_length >= seq_len:
                    total_length = (total_length // seq_len) * seq_len

                result = {
                    k: [t[i:i + seq_len] for i in range(0, total_length, seq_len)
                       ] for k, t in concatenated_examples.items()
                }
            else:
                tokenizer.pad_token = tokenizer.unk_token
                result = tokenizer(examples, padding=True, truncation=True, max_length=seq_len, return_tensors='pt')

            result["labels"] = copy.deepcopy(result["input_ids"])

            return result

        tokenized_dataset = dataset.map(partial(tokenize, mode=CONFIG['hyperparameter']['tokenize_mode']),
                                        batched=True,
                                        num_proc=16,
                                        load_from_cache_file=False,
                                        keep_in_memory=True,
                                        remove_columns='text')

        CONFIG['model']['vocab_size'] = len(tokenizer)

    def seed_worker(_):
        worker_seed = 1024
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        random.seed(worker_seed)

    train_sampler = DistributedSampler(tokenized_dataset['train'], shuffle=True) if world_size > 1 else None
    train_data = DataLoader(tokenized_dataset['train'],
                            shuffle=(train_sampler is None),
                            sampler=train_sampler,
                            drop_last=True,
                            collate_fn=default_data_collator,
                            worker_init_fn=seed_worker,
                            batch_size=CONFIG['hyperparameter']['batch_size'],
                            num_workers=4,
                            pin_memory=True)
    test_sampler = DistributedSampler(tokenized_dataset['validation'], shuffle=False) if world_size > 1 else None
    test_data = DataLoader(tokenized_dataset['validation'],
                           sampler=test_sampler,
                           collate_fn=default_data_collator,
                           worker_init_fn=seed_worker,
                           batch_size=CONFIG['hyperparameter']['batch_size'],
                           num_workers=4,
                           pin_memory=True)

    return train_data, test_data


def build_model():
    model_cfg = CONFIG['model']
    gpt2_cfg = GPT2Config(vocab_size=model_cfg['vocab_size'],
                          n_positions=model_cfg['seq_length'],
                          n_embd=model_cfg['hidden_size'],
                          n_layer=model_cfg['depth'],
                          n_head=model_cfg['num_heads'],
                          use_cache=not CONFIG['model'].get('checkpoint', False))

    model = ModelFromHF(gpt2_cfg, GPT2LMHeadModel)

    return model


class GPTLMLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


def build_loss():
    return GPTLMLoss()


def build_optimizer(params):
    optimizer = torch.optim.AdamW(params,
                                  lr=CONFIG['hyperparameter']['learning_rate'],
                                  weight_decay=CONFIG['hyperparameter']['weight_decay'])
    return optimizer


def build_scheduler(epoch_steps, optimizer):
    from transformers.optimization import get_linear_schedule_with_warmup

    max_steps = epoch_steps * CONFIG['hyperparameter']['num_epochs']
    warmup_steps = epoch_steps * CONFIG['hyperparameter']['warmup_epochs']
    lr_scheduler = get_linear_schedule_with_warmup(optimizer,
                                                   num_warmup_steps=warmup_steps,
                                                   num_training_steps=max_steps)

    return lr_scheduler


def gpt2_builder():
    model_type = CONFIG['model']['type']
    if model_type in _gpt2_configurations:
        for k, v in _gpt2_configurations[model_type].items():
            if k not in CONFIG['model']:
                CONFIG['model'][k] = v

    if 'hyperparameter' in CONFIG:
        for k, v in _default_hyperparameters.items():
            if k not in CONFIG['hyperparameter']:
                CONFIG['hyperparameter'][k] = v
    else:
        CONFIG['hyperparameter'] = _default_hyperparameters

    CONFIG['dataset'] = os.environ['DATA']
    CONFIG['tokenizer'] = os.environ['TOKENIZER']

    return build_data, build_model, build_loss, build_optimizer, build_scheduler
