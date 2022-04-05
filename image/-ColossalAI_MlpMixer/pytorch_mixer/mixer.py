import torch
import torch.nn as nn


class MlpBlock(nn.Module):
    def __init__(self, hidden_dim, mlp_dim):
        super(MlpBlock, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class MixerBlock(nn.Module):
    def __init__(self, num_tokens, hidden_dim, tokens_mlp_dim, channels_mlp_dim):
        super(MixerBlock, self).__init__()
        self.ln_token = nn.LayerNorm(hidden_dim)
        self.token_mix = MlpBlock(num_tokens, tokens_mlp_dim)
        self.ln_channel = nn.LayerNorm(hidden_dim)
        self.channel_mix = MlpBlock(hidden_dim, channels_mlp_dim)

    def forward(self, x):
        out = self.ln_token(x).transpose(1, 2)
        x = x + self.token_mix(out).transpose(1, 2)
        out = self.ln_channel(x)
        x = x + self.channel_mix(out)
        return x


class MlpMixer(nn.Module):
    def __init__(self, num_classes, num_blocks, patch_size, hidden_dim, tokens_mlp_dim, channels_mlp_dim, image_size=224):
        super(MlpMixer, self).__init__()
        num_tokens = (image_size // patch_size)**2

        self.patch_emb = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.mlp = nn.Sequential(*[MixerBlock(num_tokens, hidden_dim, tokens_mlp_dim, channels_mlp_dim) for _ in range(num_blocks)])
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.patch_emb(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.mlp(x)
        x = self.ln(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x


def mixer_s32(num_classes=1000, image_size=224, patch_size=32,**kwargs):
    return MlpMixer(num_classes, 8, patch_size, 512, 256, 2048, image_size, **kwargs)



def mixer_s16(num_classes=1000, **kwargs):
    return MlpMixer(num_classes, 8, 16, 512, 256, 2048, **kwargs)

def mixer_b32(num_classes=1000, **kwargs):
    return MlpMixer(num_classes, 12, 32, 768, 384, 3072, **kwargs)

def mixer_b16(num_classes=1000, **kwargs):
    return MlpMixer(num_classes, 12, 16, 768, 384, 3072, **kwargs)

def mixer_l32(num_classes=1000, **kwargs):
    return MlpMixer(num_classes, 24, 32, 1024, 512, 4096, **kwargs)

def mixer_l16(num_classes=1000, **kwargs):
    return MlpMixer(num_classes, 24, 16, 1024, 512, 4096, **kwargs)

def mixer_h14(num_classes=1000, **kwargs):
    return MlpMixer(num_classes, 32, 14, 1280, 640, 5120, **kwargs)
