import math
import torch.nn.functional as F
from torch.nn import Parameter
import torch
import torch.nn as nn


import colossalai
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.kernel.cuda_native.scaled_softmax import AttnMaskType
from colossalai.kernel import FusedScaleMaskSoftmax
from colossalai.context import seed
from colossalai.kernel.jit import bias_dropout_add_fused_train, bias_dropout_add_fused_inference
from colossalai.kernel.cuda_native import LayerNorm
from colossalai.utils import get_current_device


def gelu(x):
    return .5 * x * (1. + torch.erf(x / math.sqrt(2.)))

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_k)
        scores.masked_fill_(attn_mask, -1e4)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.hidden_size_per_attention_head = self.hidden_size // num_attention_heads

        self.W_Q = nn.Linear(hidden_size, self.hidden_size_per_attention_head * num_attention_heads)
        self.W_K = nn.Linear(hidden_size, self.hidden_size_per_attention_head * num_attention_heads)
        self.W_V = nn.Linear(hidden_size, self.hidden_size_per_attention_head * num_attention_heads)

    def forward(self, Q, K, V, attn_mask):
        residual, batch = Q, Q.size(0)

        per_Q = self.W_Q(Q).view(batch, -1, self.num_attention_heads, self.hidden_size_per_attention_head).transpose(1, 2)
        per_K = self.W_K(K).view(batch, -1, self.num_attention_heads, self.hidden_size_per_attention_head).transpose(1, 2)
        per_V = self.W_V(V).view(batch, -1, self.num_attention_heads, self.hidden_size_per_attention_head).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_attention_heads, 1, 1)
        # context: [batch, n_heads, seq_len, d_v]
        context = ScaledDotProductAttention(self.hidden_size_per_attention_head)(per_Q, per_K, per_V, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch, -1, self.num_attention_heads * self.hidden_size_per_attention_head)
        output = nn.Linear(self.num_attention_heads * self.hidden_size_per_attention_head, self.hidden_size, device=get_current_device(), dtype=torch.half)(context)
        return nn.LayerNorm(self.hidden_size, device=get_current_device(), dtype=torch.half)(output + residual)


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # use 4 * dimension of model to represent d_ff
        d_ff = 4 * hidden_size
        self.fc1 = nn.Linear(hidden_size, d_ff)
        self.fc2 = nn.Linear(d_ff, hidden_size)

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.fc2(gelu(self.fc1(x)))


class BertLayer(nn.Module):
    """A single transformer layer.
    Transformer layer takes input with size [b, s, h] and returns an
    output of the same size.
    """

    def __init__(self,
                 hidden_size,
                 num_attention_heads,):
        super().__init__()
        # Self attention.
        self.self_attention = MultiHeadAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
        )
        
        self.pos_ffn = PoswiseFeedForwardNet(hidden_size=hidden_size)

    def forward(self, hidden_states, attention_mask):
        outputs = self.self_attention(hidden_states, hidden_states, hidden_states, attention_mask) # enc_inputs to same Q,K,V
        outputs = self.pos_ffn(outputs) # enc_outputs: [batch_size, seq_len, d_model]
        return outputs
