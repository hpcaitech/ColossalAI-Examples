import math
import torch
from colossalai import nn as col_nn
from colossalai.registry import LAYERS, MODELS
from torch import dtype, nn

__all__ = [
    'mixer_s32',
    'mixer_s16',
    'mixer_b32',
    'mixer_b16',
    'mixer_l32',
    'mixer_l16',
    'mixer_h14',
]

_init_rules = dict(
    torch=dict(
        transformer=dict(
            weight_initializer=col_nn.init.kaiming_uniform_(a=math.sqrt(5)),
            bias_initializer=col_nn.init.xavier_uniform_(a=1, scale=1),
        ),
    ),
    jax=dict(
        transformer=dict(
            weight_initializer=col_nn.init.xavier_uniform_(),
            bias_initializer=col_nn.init.normal_(std=1e-6),
        ),
    ),
)




@LAYERS.register_module
class MlpBlock(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 mlp_dim: int,
                 dropout: float=0.,
                 dtype: dtype = None,
                 bias: bool = True,
                 init_method: str = 'torch'):
        super(MlpBlock, self).__init__()

        self.Linear = col_nn.Linear(hidden_dim,
                                    mlp_dim,
                                    dtype=dtype,
                                    bias=bias,
                                    **_init_rules[init_method]['transformer'])
        self.dropout = col_nn.Dropout(dropout)
        self.GELU = nn.GELU()
        self.Linear1 = col_nn.Linear(mlp_dim,
                                    hidden_dim,
                                    dtype=dtype,
                                    bias=bias,
                                    **_init_rules[init_method]['transformer'])

    def forward(self, x):
        x = self.Linear(x)
        x = self.GELU(x)
        x = self.Linear1(x)
        return x


@LAYERS.register_module
class MixerBlock(nn.Module):
    def __init__(self,
                 num_tokens: int,
                 hidden_dim: int,
                 tokens_mlp_dim: int,
                 channels_mlp_dim: int,
                 layernorm_epsilon: float = 1e-6,
                 dtype: dtype = None):
        super(MixerBlock, self).__init__()
        self.ln_token = col_nn.LayerNorm(normalized_shape=hidden_dim, eps=layernorm_epsilon, dtype=dtype)
        self.token_mix = MlpBlock(num_tokens, tokens_mlp_dim)
        self.ln_channel = col_nn.LayerNorm(normalized_shape=hidden_dim, eps=layernorm_epsilon, dtype=dtype)
        self.channel_mix = MlpBlock(hidden_dim, channels_mlp_dim)


    def forward(self, x):
        out = self.ln_token(x).transpose(1, 2)
        x = x + self.token_mix(out).transpose(1, 2)
        out = self.ln_channel(x)
        x = x + self.channel_mix(out)
        return x


@LAYERS.register_module
class MlpMixer(nn.Module):
    def __init__(self,
                 num_classes: int,
                 num_blocks: int,
                 patch_size: int,
                 hidden_dim: int,
                 tokens_mlp_dim: int,
                 channels_mlp_dim: int,
                 image_size=224,
                 layernorm_epsilon: float = 1e-6,
                 dtype: dtype = None,
                 bias: bool = True,
                 init_method: str = 'torch'):
        super(MlpMixer, self).__init__()
        num_tokens = (image_size // patch_size)**2

        self.patch_emb = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size, bias=False)

        self.mlp = nn.Sequential(*[MixerBlock(num_tokens, hidden_dim, tokens_mlp_dim, channels_mlp_dim,layernorm_epsilon,dtype) for _ in range(num_blocks)])
        self.ln = col_nn.LayerNorm(normalized_shape=hidden_dim, eps=layernorm_epsilon, dtype=dtype)

        self.fc =col_nn.Linear(hidden_dim,
                                    num_classes,
                                    dtype=dtype,
                                    bias=bias,
                                    **_init_rules[init_method]['transformer'])


    def forward(self, x):
        x = self.patch_emb(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.mlp(x)
        x = self.ln(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x





@MODELS.register_module
def mixer_s32(num_classes=1000, image_size=224, patch_size=32,**kwargs):
    return MlpMixer(num_classes, 8, patch_size, 512, 256, 2048, image_size, **kwargs)


@MODELS.register_module
def mixer_s16(num_classes=1000, image_size=224, patch_size=32,**kwargs):
    return MlpMixer(num_classes, 8, patch_size, 16, 512, 256, 2048, **kwargs)


@MODELS.register_module
def mixer_b32(num_classes=1000, image_size=224, patch_size=32,**kwargs):
    return MlpMixer(num_classes, 12, patch_size, 32, 768, 384, 3072, **kwargs)


@MODELS.register_module
def mixer_b16(num_classes=1000, image_size=224, patch_size=32,**kwargs):
    return MlpMixer(num_classes, 12, patch_size, 16, 768, 384, 3072, **kwargs)


@MODELS.register_module
def mixer_l32(num_classes=1000,image_size=224, patch_size=32,**kwargs):
    return MlpMixer(num_classes, 24, patch_size, 32, 1024, 512, 4096, **kwargs)



@MODELS.register_module
def mixer_l16(num_classes=1000, image_size=224, patch_size=32,**kwargs):
    return MlpMixer(num_classes, 24, patch_size, 16, 1024, 512, 4096, **kwargs)



@MODELS.register_module
def mixer_h14(num_classes=1000, image_size=224, patch_size=32,**kwargs):
    return MlpMixer(num_classes, 32, patch_size, 14, 1280, 640, 5120, **kwargs)






































