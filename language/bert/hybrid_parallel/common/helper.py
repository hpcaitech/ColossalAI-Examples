import os

import torch
from torch.distributed import get_world_size
from transformers import BertConfig, BertTokenizer

from zero.common.utils import CONFIG, ModelFromHF, get_model_size
from bert.colossalai_utils.model_zoo.colo_bert import ColoBertMaskedLMLoss, ColoBertForMaskedLM, create_colo_bert_pipeline_model

_bert_base = dict(
    seq_length=512,
    vocab_size=50304,
    hidden_size=768,
    num_heads=12,
    depth=12,
    ff_size=3072,
    checkpoint=False,
    evaluation='ppl',
)

_bert_large = dict(
    seq_length=512,
    vocab_size=50304,
    hidden_size=1024,
    num_heads=16,
    depth=24,
    ff_size=3072,
    checkpoint=False,
    evaluation='ppl',
)

_bert_configurations = dict(bert=_bert_base, bert_base=_bert_base, bert_large=_bert_large)

_default_hyperparameters = dict(
    tokenize_mode='concat',
    batch_size=8,
    learning_rate=5e-5,
    weight_decay=1e-2,
    num_epochs=2,
    warmup_epochs=1,
    steps_per_epoch=100,
)


def build_data():
    import random
    from functools import partial
    from itertools import chain

    import numpy as np
    from datasets import load_from_disk, set_progress_bar_enabled
    from torch.utils.data import DataLoader, DistributedSampler
    from transformers import DataCollatorForLanguageModeling

    world_size = get_world_size()

    set_progress_bar_enabled(False)
    dataset = load_from_disk(CONFIG['dataset'])
    tokenizer = BertTokenizer(vocab_file=CONFIG['tokenizer'] + '/vocab.txt')

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

        return result

    tokenized_dataset = dataset.map(partial(tokenize, mode=CONFIG['hyperparameter']['tokenize_mode']),
                                    batched=True,
                                    num_proc=16,
                                    load_from_cache_file=False,
                                    keep_in_memory=True,
                                    remove_columns='text')

    def seed_worker(_):
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
                            batch_size=CONFIG['hyperparameter']['batch_size'],
                            pin_memory=True)
    test_sampler = DistributedSampler(tokenized_dataset['validation'], shuffle=False) if world_size > 1 else None
    test_data = DataLoader(tokenized_dataset['validation'],
                           sampler=test_sampler,
                           drop_last=True,
                           collate_fn=data_collator,
                           worker_init_fn=seed_worker,
                           batch_size=CONFIG['hyperparameter']['batch_size'],
                           pin_memory=True)

    return train_data, test_data


def build_model():
    model_cfg = CONFIG['model']
    bert_cfg = BertConfig(vocab_size=model_cfg['vocab_size'],
                          hidden_size=model_cfg['hidden_size'],
                          num_hidden_layers=model_cfg['depth'],
                          num_attention_heads=model_cfg['num_heads'],
                          intermediate_size=model_cfg['ff_size'],
                          max_position_embeddings=model_cfg['seq_length'],
                          use_cache=not CONFIG['model'].get('checkpoint', False))

    use_pipeline = 'parallel' in CONFIG and 'pipeline' in CONFIG['parallel'] and int(CONFIG['parallel']['pipeline']) > 1
    if use_pipeline:
        model = create_colo_bert_pipeline_model(bert_cfg)
    else:
        model = ModelFromHF(bert_cfg, ColoBertForMaskedLM)

    return model


def build_loss():
    return ColoBertMaskedLMLoss()


def build_optimizer(params):
    optimizer = torch.optim.AdamW(params,
                                  lr=CONFIG['hyperparameter']['learning_rate'],
                                  weight_decay=CONFIG['hyperparameter']['weight_decay'])
    return optimizer


def build_scheduler(epoch_steps, optimizer):
    from transformers.optimization import get_linear_schedule_with_warmup
    from colossalai.nn.lr_scheduler import LinearWarmupLR

    max_steps = epoch_steps * CONFIG['hyperparameter']['num_epochs']
    warmup_steps = epoch_steps * CONFIG['hyperparameter']['warmup_epochs']

    if CONFIG['method'] == 'colossalai':
        lr_scheduler = LinearWarmupLR(optimizer, total_steps=max_steps, warmup_steps=warmup_steps)
    else:
        lr_scheduler = get_linear_schedule_with_warmup(optimizer,
                                                       num_warmup_steps=warmup_steps,
                                                       num_training_steps=max_steps)

    return lr_scheduler


def bert_builder():
    model_type = CONFIG['model']['type']
    if model_type in _bert_configurations:
        for k, v in _bert_configurations[model_type].items():
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
    if 'numel' not in CONFIG['model']:
        CONFIG['model']['numel'] = get_model_size(build_model())

    return build_data, build_model, build_loss, build_optimizer, build_scheduler
