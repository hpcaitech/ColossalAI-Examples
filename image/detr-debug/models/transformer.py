import copy

import torch
import torch.nn.functional as F
from torch import nn
import math
from colossalai.registry import LAYERS, MODELS
from colossalai import nn as col_nn

@MODELS.register_module
class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        encoder_norm = col_nn.LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        decoder_norm = col_nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm, return_intermediate=return_intermediate_dec)


        self.d_model = d_model
        self.nhead = nhead

    def forward(self, src, mask, query_embed, pos_embed):
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        # mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, pos=pos_embed)

        hs = self.decoder(tgt, memory, pos=pos_embed, query_pos=query_embed)

        return hs.transpose(1, 2)

@LAYERS.register_module
class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, pos):
        output = src if pos is None else (src + pos)
        output = output.transpose(0, 1)

        for layer in self.layers:
            output = layer(output)

        if self.norm is not None:
            output = self.norm(output)

        return output

@LAYERS.register_module
class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory, pos, query_pos):
        intermediate = []

        for layer in self.layers:
            tgt = layer(tgt, memory, pos=pos, query_pos=query_pos).transpose(0, 1)

            if self.return_intermediate:
                intermediate.append(self.norm(tgt))

        return torch.stack(intermediate)

@LAYERS.register_module
class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.selfAttn = MultiHeadAttention(d_model, dim_feedforward, nhead, dropout)
        self.feedForward = FeedForward(d_model, dim_feedforward, dropout)

        self.norm_1 = col_nn.LayerNorm(d_model)
        self.norm_2 = col_nn.LayerNorm(d_model)
        self.dropout_1 = col_nn.Dropout(dropout)
        self.dropout_2 = col_nn.Dropout(dropout)

    def forward(self, x):
        x1 = self.norm_1(x)
        x = x + self.dropout_1(self.selfAttn(x1, x1, x1))
        x2 = self.norm_2(x)
        out = x + self.dropout_2(self.feedForward(x2))
        return out


@LAYERS.register_module
class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.selfAttn = MultiHeadAttention(d_model, dim_feedforward, nhead, dropout)

        self.linear_1 = col_nn.Linear(d_model, dim_feedforward)
        self.linear_2 = col_nn.Linear(dim_feedforward, d_model)
        self.norm_1 = col_nn.LayerNorm(d_model)
        self.norm_2 = col_nn.LayerNorm(d_model)
        self.norm_3 = col_nn.LayerNorm(d_model)
        self.dropout_1 = col_nn.Dropout(dropout)
        self.dropout_2 = col_nn.Dropout(dropout)
        self.dropout_3 = col_nn.Dropout(dropout)
        self.dropout_4 = col_nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, pos, query_pos):
        tgt = tgt.transpose(0, 1)
        query_pos = query_pos.transpose(0, 1)
        pos = pos.transpose(0, 1)

        q = k = self.with_pos_embed(tgt, query_pos)

        tgt2 = self.selfAttn(q, k, tgt)

        tgt = tgt + self.dropout_1(tgt2)
        tgt = self.norm_1(tgt)
        tgt2 = self.selfAttn(q, self.with_pos_embed(memory, pos), memory)
        tgt = tgt + self.dropout_2(tgt2)
        tgt = self.norm_2(tgt)
        tgt2 = self.linear_2(self.dropout_3(F.relu(self.linear_1(tgt))))
        tgt = tgt + self.dropout_4(tgt2)
        tgt = self.norm_3(tgt)
        return tgt

def transpose_qkv(X, num_heads):
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])

def transpose_output(X, num_heads):
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

@LAYERS.register_module
class SelfAttention(nn.Module):
    def __init__(self, dropout,):
        super(SelfAttention, self).__init__()
        self.dropout = col_nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = torch.softmax(scores, dim=2)
        return torch.bmm(self.dropout(self.attention_weights), values)

@LAYERS.register_module
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_hiddens,
                 num_heads, dropout, bias=False):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention = SelfAttention(dropout)
        self.W_q = col_nn.Linear(d_model, num_hiddens, bias=bias)
        self.W_k = col_nn.Linear(d_model, num_hiddens, bias=bias)
        self.W_v = col_nn.Linear(d_model, num_hiddens, bias=bias)
        self.W_o = col_nn.Linear(num_hiddens, d_model, bias=bias)

    def forward(self, queries, keys, values):
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        output = self.attention(queries, keys, values)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)

@LAYERS.register_module
class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout):
        super().__init__()
        self.linear_1 = col_nn.Linear(d_model, dim_feedforward)
        self.ff_drop = col_nn.Dropout(dropout)
        self.linear_2 = col_nn.Linear(dim_feedforward, d_model)
    def forward(self, x):
        x = self.ff_drop(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        return_intermediate_dec=True,
    )
