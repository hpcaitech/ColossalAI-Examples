from colossalai.context.parallel_mode import ParallelMode
import math
import torch.nn as nn
import torch
from .deepnet_configs import DeepNetTransformerLayer1D, FusedDeepNetTransformerLayer1D
from .embed import HiddenParallelEmbedding, HiddenParallelLMHead1D, VocabParallelEmbedding, VocabParallelLMHead1D
from colossalai.nn.layer.wrapper import PipelineSharedModuleWrapper
from colossalai.core import global_context as gpc
import inspect
from colossalai.builder.pipeline import partition_uniform
from colossalai import kernel
from colossalai.logging import get_dist_logger


class GenericPipelineDeepNet(nn.Module):
    def __init__(self, embedding=None, blocks=None, norm=None, head=None) -> None:
        super().__init__()
        self.embedding = embedding
        self.blocks = blocks
        self.norm = norm
        self.head = head
        assert blocks is not None
        if norm is not None or head is not None:
            assert norm is not None and head is not None

    def forward(self, hidden_states=None, input_ids=None, attention_mask=None):
        if self.embedding is not None:
            hidden_states = self.embedding(input_ids=input_ids)
        batch_size = hidden_states.shape[0]
        attention_mask = attention_mask.view(batch_size, -1)
        attention_mask = attention_mask[:, None, None, :]
        attention_mask = attention_mask.to(dtype=hidden_states.dtype)  # fp16 compatibility
        attention_mask = (1.0 - attention_mask) * -10000.0
        for block in self.blocks:
            hidden_states, attention_mask = block(hidden_states, attention_mask)
        if self.norm is not None:
            hidden_states = self.head(self.norm(hidden_states))
        return hidden_states


class PipelineDeepNet1D(GenericPipelineDeepNet):
    def __init__(self,
                 num_layers: int = 12,
                 hidden_size: int = 768,
                 num_attention_heads: int = 12,
                 vocab_size: int = 50304,
                 embed_drop_rate: float = 0.,
                 act_func: str = 'gelu',
                 mlp_ratio: int = 4.0,
                 attn_drop_rate: float = 0.,
                 drop_rate: float = 0.,
                 dtype: torch.dtype = torch.float,
                 checkpoint: bool = False,
                 max_position_embeddings: int = 1024,
                 layer_norm_epsilon: float = 1e-5,
                 first: bool = False,
                 last: bool = False,
                 embed_split_hidden=False):
        embedding = None
        norm = None
        head = None
        alpha = math.sqrt(2*num_layers)
        embed_cls = VocabParallelEmbedding
        head_cls = VocabParallelLMHead1D
        if embed_split_hidden:
            embed_cls = HiddenParallelEmbedding
            head_cls = HiddenParallelLMHead1D
        if first:
            embedding = embed_cls(hidden_size, vocab_size, max_position_embeddings, embed_drop_rate, dtype=dtype)
        blocks = nn.ModuleList([
            DeepNetTransformerLayer1D(hidden_size, num_attention_heads, act_func=act_func, mlp_ratio=mlp_ratio, attention_dropout_prob=attn_drop_rate,
                                  hidden_dropout_prob=drop_rate, dtype=dtype, checkpoint=checkpoint, max_position_embeddings=max_position_embeddings,
                                  layer_norm_epsilon=layer_norm_epsilon, alpha=alpha)
            for _ in range(num_layers)
        ])
        if last:
            norm = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
            head = head_cls(vocab_size=vocab_size,
                            embed_dim=hidden_size,
                            dtype=dtype)
        super().__init__(embedding=embedding, blocks=blocks, norm=norm, head=head)


class FusedPipelineDeepNet1D(GenericPipelineDeepNet):
    def __init__(self,
                 num_layers: int = 12,
                 hidden_size: int = 768,
                 num_attention_heads: int = 12,
                 vocab_size: int = 50304,
                 embed_drop_rate: float = 0.,
                 act_func: str = 'gelu',
                 mlp_ratio: int = 4.0,
                 attn_drop_rate: float = 0.,
                 drop_rate: float = 0.,
                 dtype: torch.dtype = torch.float,
                 checkpoint: bool = False,
                 max_position_embeddings: int = 1024,
                 layer_norm_epsilon: float = 1e-5,
                 first: bool = False,
                 last: bool = False,
                 embed_split_hidden=False):
        embedding = None
        norm = None
        head = None
        alpha = math.sqrt(2 * num_layers)
        embed_cls = VocabParallelEmbedding
        head_cls = VocabParallelLMHead1D
        if embed_split_hidden:
            embed_cls = HiddenParallelEmbedding
            head_cls = HiddenParallelLMHead1D
        if first:
            embedding = embed_cls(hidden_size, vocab_size, max_position_embeddings, embed_drop_rate, dtype=dtype)
        blocks = nn.ModuleList([
            FusedDeepNetTransformerLayer1D(hidden_size, num_attention_heads, act_func=act_func, mlp_ratio=mlp_ratio, attention_dropout_prob=attn_drop_rate,
                                       hidden_dropout_prob=drop_rate, dtype=dtype, checkpoint=checkpoint, max_position_embeddings=max_position_embeddings,
                                       layer_norm_epsilon=layer_norm_epsilon, alpha=alpha)
            for _ in range(num_layers)
        ])
        if last:
            norm = kernel.LayerNorm(hidden_size, eps=layer_norm_epsilon)
            head = head_cls(vocab_size=vocab_size,
                            embed_dim=hidden_size,
                            dtype=dtype)
        super().__init__(embedding=embedding, blocks=blocks, norm=norm, head=head)

    def forward(self, hidden_states=None, input_ids=None, attention_mask=None):
        if self.embedding is not None:
            hidden_states = self.embedding(input_ids=input_ids)
        attention_mask = attention_mask.to(dtype=hidden_states.dtype)  # fp16 compatibility
        for block in self.blocks:
            hidden_states, attention_mask = block(hidden_states, attention_mask)
        if self.norm is not None:
            hidden_states = self.head(self.norm(hidden_states))
        return hidden_states


def _filter_kwargs(func, kwargs):
    sig = inspect.signature(func)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def _build_generic_deepnet_pipeline_1d(module_cls, num_layers, num_chunks, device=torch.device('cuda'), **kwargs):
    logger = get_dist_logger()
    pipeline_size = gpc.get_world_size(ParallelMode.PIPELINE)
    pipeline_rank = gpc.get_local_rank(ParallelMode.PIPELINE)
    rank = gpc.get_global_rank()
    wrapper = PipelineSharedModuleWrapper([0, pipeline_size - 1])
    parts = partition_uniform(num_layers, pipeline_size, num_chunks)[pipeline_rank]
    models = []
    for start, end in parts:
        kwargs['num_layers'] = end - start
        kwargs['first'] = start == 0
        kwargs['last'] = end == num_layers
        logger.info(f'Rank{rank} build layer {start}-{end}, {end-start}/{num_layers} layers')
        chunk = module_cls(**_filter_kwargs(module_cls.__init__, kwargs)).to(device)
        if start == 0:
            wrapper.register_module(chunk.embedding.word_embeddings)
        elif end == num_layers:
            wrapper.register_module(chunk.head)
        models.append(chunk)
    if len(models) == 1:
        model = models[0]
    else:
        model = nn.ModuleList(models)

    numel = 0
    for _, param in model.named_parameters(recurse=True):
        numel += param.numel()
    logger.info(f'Rank{rank}/{gpc.get_local_rank(ParallelMode.PIPELINE)} model size = {numel * 2 / 1e9} GB')
    return model


def _build_deepnet_pipeline_1d(num_layers, num_chunks, device=torch.device('cuda'), fused=False, **kwargs):
    model = FusedPipelineDeepNet1D if fused else PipelineDeepNet1D
    return _build_generic_deepnet_pipeline_1d(model, num_layers, num_chunks, device, **kwargs)


def deepnet_small_pipeline_1D(num_chunks=1, checkpoint=False, dtype=torch.float, embed_split_hidden=False, fused=False):
    cfg = dict(hidden_size=768, num_attention_heads=12, checkpoint=checkpoint,
               dtype=dtype, embed_split_hidden=embed_split_hidden)
    return _build_deepnet_pipeline_1d(12, num_chunks, fused=fused, **cfg)
