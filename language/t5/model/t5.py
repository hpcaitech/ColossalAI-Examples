#from t5_lm_head import T5LMHeadModel
from transformers import T5ForConditionalGeneration, T5Config
import torch.nn as nn
from colossalai.logging import get_dist_logger

__all__ = [
    "T5LMModel",
    "T5LMLoss",
    "T5_small",
    "T5_base",
    "T5_large",
    "T5_3b",
    "T5_11b",
]

class T5LMModel(nn.Module):
    def __init__(self, d_model=512, d_kv=64, d_ff=2048, num_layers=6, num_heads=8, decoder_start_token_id=0, pad_token_id=0, return_dict=False, checkpoint=False):
        super().__init__()
        self.checkpoint = checkpoint
        self.model = T5ForConditionalGeneration(
            T5Config(d_model=d_model, d_kv=d_kv, d_ff=d_ff, num_layers=num_layers, num_heads=num_heads, decoder_start_token_id=decoder_start_token_id, pad_token_id=pad_token_id, return_dict=return_dict))

    def forward(self, input_ids, labels):
        # Only return lm_logits
        logger = get_dist_logger()
        output = self.model(input_ids=input_ids, labels=labels, use_cache=not self.checkpoint)
        return output[1]


class T5LMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


def T5_small(checkpoint=False):
    cfg = dict(d_model=512, d_ff=2048, d_kv=64, num_layers=6, num_heads=8)
    return T5LMModel(checkpoint=checkpoint, **cfg)


def T5_base(checkpoint=False):
    cfg = dict(d_model=768, d_ff=3072, d_kv=64, num_layers=12, num_heads=12)
    return T5LMModel(checkpoint=checkpoint, **cfg)


def T5_large(checkpoint=False):
    cfg = dict(d_model=1024, d_ff=4096, d_kv=64, num_layers=24, num_heads=12)
    return T5LMModel(checkpoint=checkpoint, **cfg)


def T5_3b(checkpoint=False):
    cfg = dict(d_model=1024, d_ff=16384, d_kv=128, num_layers=24, num_heads=32)
    return T5LMModel(checkpoint=checkpoint, **cfg)


def T5_11b(checkpoint=False):
    cfg = dict(d_model=1024, d_ff=65536, d_kv=128, num_layers=24, num_heads=128)
    return T5LMModel(checkpoint=checkpoint, **cfg)
