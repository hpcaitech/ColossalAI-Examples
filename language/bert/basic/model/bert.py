from colossalai.context.parallel_mode import ParallelMode
import torch
import torch.nn as nn
import inspect
from .layers import Embedding, BertLayer, BertDualHead, PreProcessor, VocabEmbedding
from .layers.init_method import init_normal, output_init_normal
from colossalai.core import global_context as gpc
from colossalai.context import ParallelMode
from colossalai.kernel import LayerNorm
from colossalai.nn.layer.wrapper import PipelineSharedModuleWrapper
from colossalai.logging import get_dist_logger
from colossalai.builder.pipeline import partition_uniform


def get_attn_pad_mask(seq_q, seq_k):
    batch_size, seq_len = seq_q.size()
    pad_attn_mask = seq_q.data.eq(0).unsqueeze(1)
    pad_attn_mask = pad_attn_mask.expand(batch_size, seq_len, seq_len)
    return pad_attn_mask


class BertForPretrain(nn.Module):

    def __init__(self,
                 vocab_size,
                 hidden_size,
                 max_sequence_length,
                 num_attettion_heads,
                 num_layers,
                 add_binary_head,
                 is_naive_fp16,
                 num_tokentypes=2,
                 dropout_prob=0.1,
                 mlp_ratio=4,
                 init_std=0.02,
                 convert_fp16_to_fp32_in_softmax=False,
                 ):
        super().__init__()
        self.init_std = init_std
        self.num_layers = num_layers

        if not add_binary_head:
            num_tokentypes = 0

        self.preprocessor = PreProcessor()
        self.embedding = Embedding(hidden_size=hidden_size,
                                   vocab_size=vocab_size,
                                   max_sequence_length=max_sequence_length,
                                   embedding_dropout_prob=dropout_prob,
                                   num_tokentypes=num_tokentypes)
        self.bert_layers = nn.ModuleList()

        for i in range(num_layers):
            bert_layer = BertLayer(hidden_size=hidden_size, num_attention_heads=num_attettion_heads)
            self.bert_layers.append(bert_layer)

        self.layer_norm = LayerNorm(hidden_size)
        self.head = BertDualHead(hidden_size, self.embedding.word_embedding_weight.size(0),
                                 add_binary_head=add_binary_head)

    def forward(self, input_ids, attention_masks, tokentype_ids, lm_labels):
        pos_ids, attention_masks = self.preprocessor(input_ids, attention_masks)

        hidden_states = self.embedding(input_ids, pos_ids, tokentype_ids)
        
        attention_masks = get_attn_pad_mask(input_ids, input_ids)

        for idx, layer in enumerate(self.bert_layers):
            hidden_states = layer(hidden_states, attention_masks)

        hidden_states = hidden_states.transpose(0, 1).contiguous()
        output = self.layer_norm(hidden_states)

        # word_embedding: [vocab_size, hidden_size]
        return self.head(output, self.embedding.word_embedding_weight, lm_labels)


class PipelineBertForPretrain(nn.Module):

    def __init__(self,
                 vocab_size,
                 hidden_size,
                 max_sequence_length,
                 num_attettion_heads,
                 num_layers,
                 add_binary_head,
                 is_naive_fp16,
                 num_tokentypes=2,
                 dropout_prob=0.1,
                 mlp_ratio=4,
                 init_std=0.02,
                 convert_fp16_to_fp32_in_softmax=False,
                 first_stage=True,
                 last_stage=True,
                 start_idx=None,
                 end_idx=None):
        super().__init__()
        self.init_std = init_std
        self.num_layers = num_layers

        if not add_binary_head:
            num_tokentypes = 0

        self.first_stage = first_stage
        self.last_stage = last_stage

        self.preprocessor = PreProcessor()

        if self.first_stage:
            self.embedding = Embedding(hidden_size=hidden_size,
                                       vocab_size=vocab_size,
                                       max_sequence_length=max_sequence_length,
                                       embedding_dropout_prob=dropout_prob,
                                       num_tokentypes=num_tokentypes)

        # transformer layers
        self.bert_layers = nn.ModuleList()

        if start_idx is None and end_idx is None:
            start_idx = 0
            end_idx = num_layers

        for i in range(start_idx, end_idx):
            bert_layer = BertLayer(hidden_size=hidden_size, num_attention_heads=num_attettion_heads)
            self.bert_layers.append(bert_layer)

        if self.last_stage:
            self.word_embeddings = VocabEmbedding(vocab_size, hidden_size)
            self.layer_norm = LayerNorm(hidden_size)
            self.head = BertDualHead(hidden_size, vocab_size,
                                     add_binary_head=add_binary_head)

    def forward(self, input_ids, attention_masks, tokentype_ids, lm_labels):
        if self.first_stage:
            pos_ids, _ = self.preprocessor(input_ids, attention_masks)

        if self.first_stage:
            hidden_states = self.embedding(input_ids, pos_ids, tokentype_ids)
            hidden_states = hidden_states.transpose(0, 1).contiguous()
        else:
            hidden_states = input_ids
            
        attention_masks = get_attn_pad_mask(input_ids, input_ids)

        for idx, layer in enumerate(self.bert_layers):
            hidden_states = layer(hidden_states, attention_masks)

        output = hidden_states

        return output


def _filter_kwargs(func, kwargs):
    sig = inspect.signature(func)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def build_pipeline_bert(num_layers, num_chunks, device=torch.device('cuda'), **kwargs):
    logger = get_dist_logger()
    pipeline_size = gpc.get_world_size(ParallelMode.PIPELINE)
    pipeline_rank = gpc.get_local_rank(ParallelMode.PIPELINE)
    rank = gpc.get_global_rank()
    wrapper = PipelineSharedModuleWrapper([0, pipeline_size - 1])
    parts = partition_uniform(num_layers, pipeline_size, num_chunks)[pipeline_rank]
    models = []
    for start, end in parts:
        kwargs['num_layers'] = num_layers
        kwargs['start_idx'] = start
        kwargs['end_idx'] = end
        kwargs['first_stage'] = start == 0
        kwargs['last_stage'] = end == num_layers
        logger.info(f'Rank{rank} build layer {start}-{end}, {end-start}/{num_layers} layers')
        chunk = PipelineBertForPretrain(**_filter_kwargs(PipelineBertForPretrain.__init__, kwargs)).to(device)
        if start == 0:
            wrapper.register_module(chunk.embedding.word_embeddings)
        elif end == num_layers:
            wrapper.register_module(chunk.word_embeddings)
        models.append(chunk)
    if len(models) == 1:
        model = models[0]
    else:
        model = nn.ModuleList(models)
    return model
