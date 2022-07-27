from concurrent.futures import process
import transformers
import logging
import torch
from colossalai.nn.lr_scheduler import LinearWarmupLR
from titans.dataloader.bert import get_bert_pretrain_data_loader
from transformers import BertForPreTraining
from colossalai.nn.optimizer import HybridAdam
from pathlib import Path
import torch.distributed as dist

__all__ = ['get_model', 'get_optimizer', 'get_lr_scheduler', 'get_dataloader_for_pretraining']


def get_model(config_file):
    config = transformers.BertConfig.from_json_file(config_file)
    model = BertForPreTraining(config)
    return config, model


def get_optimizer(model, lr):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']

    # configure the weight decay for bert models
    optimizer_grouped_parameters = [{
        'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay': 0.01
    }, {
        'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay': 0.0
    }]

    optimizer = HybridAdam(optimizer_grouped_parameters, lr=lr)
    return optimizer


def get_lr_scheduler(optimizer, total_steps, warmup_ratio):
    warmup_steps = int(total_steps * warmup_ratio)
    lr_scheduler = LinearWarmupLR(optimizer, total_steps=total_steps, warmup_steps=warmup_steps)
    return lr_scheduler


def get_dataloader_for_pretraining(root,
                                   vocab_file,
                                   global_batch_size,
                                   process_group=None,
                                   num_workers=4,
                                   log_dir='./logs',
                                   seed=1024,
                                   epoch=0):
    dataloader = get_bert_pretrain_data_loader(
        root,
        vocab_file=vocab_file,
        process_group=process_group,
        data_loader_kwargs={
            'batch_size': global_batch_size,
            'num_workers': num_workers,
            'pin_memory': True,
        },
        base_seed=seed,
        log_dir=log_dir,
        log_level=logging.WARNING,
        start_epoch=epoch,
    )

    return dataloader


def save_checkpoint(model, output_path, prefix):

    if isinstance(output_path, str):
        output_path = Path(output_path)

    model_to_save = model
    state_dict = model_to_save.state_dict()

    if dist.get_rank() == 0:
        torch.save(
            {"model": state_dict},
            output_path.joinpath(f'{prefix}_model_weights.pth'),
        )