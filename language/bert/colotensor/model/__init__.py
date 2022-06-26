from language.bert.colotensor.model.hfmodel import ModelFromHF
from colossalai.core import global_context as gpc
from transformers import BertConfig, BertForMaskedLM

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

_bert_configurations = dict(
    bert=_bert_base,
    bert_base=_bert_base,
    bert_large=_bert_large
)

def build_model():
    model_cfg = _bert_configurations[gpc.config.model.type]
    bert_cfg = BertConfig(vocab_size=model_cfg['vocab_size'],
                          hidden_size=model_cfg['hidden_size'],
                          num_hidden_layers=model_cfg['depth'],
                          num_attention_heads=model_cfg['num_heads'],
                          intermediate_size=model_cfg['ff_size'],
                          max_position_embeddings=model_cfg['seq_length'],
                          use_cache=not gpc.config.model.get('checkpoint', False))

    model = ModelFromHF(bert_cfg, BertForMaskedLM)

    return model

__all__ = ["build_model"]