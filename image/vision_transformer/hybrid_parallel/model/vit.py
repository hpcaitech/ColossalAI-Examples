import math
from typing import Callable

import inspect
import torch
from colossalai import nn as col_nn
from colossalai.registry import LAYERS, MODELS
from colossalai.logging import get_dist_logger
from colossalai.core import global_context as gpc
from colossalai.context import ParallelMode
from colossalai.builder.pipeline import partition_uniform
from torch import dtype, nn
from model_zoo.vit.vit import ViTBlock, ViTEmbedding, ViTHead


@MODELS.register_module
class PipelineVisionTransformer(nn.Module):
    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_chans: int = 3,
                 num_classes: int = 1000,
                 depth: int = 12,
                 num_heads: int = 12,
                 dim: int = 768,
                 mlp_ratio: int = 4,
                 attention_dropout: float = 0.,
                 dropout: float = 0.1,
                 drop_path: float = 0.,
                 layernorm_epsilon: float = 1e-6,
                 activation: Callable = nn.functional.gelu,
                 representation_size: int = None,
                 dtype: dtype = None,
                 bias: bool = True,
                 checkpoint: bool = False,
                 init_method: str = 'torch',
                 first_stage=True,
                 last_stage=True,
                 start_idx=None,
                 end_idx=None,):
        super().__init__()

        layers = []

        if first_stage:
            embed = ViTEmbedding(img_size=img_size,
                                 patch_size=patch_size,
                                 in_chans=in_chans,
                                 embedding_dim=dim,
                                 dropout=dropout,
                                 dtype=dtype,
                                 init_method=init_method)
            layers.append(embed)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]

        if start_idx is None and end_idx is None:
            start_idx = 0
            end_idx = depth

        blocks = [
            ViTBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                attention_dropout=attention_dropout,
                dropout=dropout,
                drop_path=dpr[i],
                activation=activation,
                dtype=dtype,
                bias=bias,
                checkpoint=checkpoint,
                init_method=init_method,
            ) for i in range(start_idx, end_idx)
        ]
        layers.extend(blocks)

        if last_stage:
            norm = col_nn.LayerNorm(normalized_shape=dim, eps=layernorm_epsilon, dtype=dtype)
            head = ViTHead(dim=dim,
                           num_classes=num_classes,
                           representation_size=representation_size,
                           dtype=dtype,
                           bias=bias,
                           init_method=init_method)
            layers.extend([norm, head])

        self.layers = nn.Sequential(
            *layers
        )

    def forward(self, x):
        x = self.layers(x)
        return x


def _filter_kwargs(func, kwargs):
    sig = inspect.signature(func)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def _build_pipeline_vit(module_cls, num_layers, num_chunks, device=torch.device('cuda'), **kwargs):
    logger = get_dist_logger()
    if gpc.is_initialized(ParallelMode.PIPELINE):
        pipeline_size = gpc.get_world_size(ParallelMode.PIPELINE)
        pipeline_rank = gpc.get_local_rank(ParallelMode.PIPELINE)
    else:
        pipeline_size = 1
        pipeline_rank = 0
    rank = gpc.get_global_rank()
    parts = partition_uniform(num_layers, pipeline_size, num_chunks)[pipeline_rank]
    models = []

    for start, end in parts:
        kwargs['first_stage'] = start == 0
        kwargs['last_stage'] = end == num_layers
        kwargs['start_idx'] = start
        kwargs['end_idx'] = end
        logger.info(f'Rank{rank} build layer {start}-{end}, {end-start}/{num_layers} layers')
        chunk = module_cls(**_filter_kwargs(module_cls.__init__, kwargs)).to(device)
        models.append(chunk)
    if len(models) == 1:
        model = models[0]
    else:
        model = nn.ModuleList(models)
    return model


def build_pipeline_vit(num_layers, num_chunks, device=torch.device('cuda'), **kwargs):
    return _build_pipeline_vit(PipelineVisionTransformer, num_layers, num_chunks, device, **kwargs)
