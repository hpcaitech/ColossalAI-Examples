from typing import Callable
import torch
from colossalai import nn as col_nn
from colossalai.registry import MODELS
from torch import dtype, nn
from model_zoo.vit.vit import ViTBlock, ViTEmbedding
from utils import heads, objectives
import torch.nn.functional as F
from colossalai.nn.layer.colossalai_layer import LayerNorm
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings


@MODELS.register_module
class ViLT(nn.Module):
    def __init__(
            self,
            config,
            img_size: int = 384,
            patch_size: int = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            depth: int = 12,
            num_heads: int = 12,
            dim: int = 768,
            mlp_ratio: int = 4,
            attention_dropout: float = 0.,
            dropout: float = 0.1,
            dropout_prob=0.1,
            drop_path: float = 0.,
            init_std=0.02,
            layernorm_epsilon: float = 1e-6,
            activation: Callable = nn.functional.gelu,
            representation_size: int = None,
            convert_fp16_to_fp32_in_softmax=False,
            dtype: dtype = None,
            bias: bool = True,
            checkpoint: bool = False,
            init_method: str = 'torch',
            first_stage=True,
            last_stage=True,
            start_idx=0,
            end_idx=None,):

        super().__init__()
        max_sequence_length = config["max_text_len"]
        num_layers = config["num_layers"]
        vocab_size = config["vocab_size"]
        self.vocab_size = vocab_size
        hidden_size = config["hidden_size"]
        self.first_stage = first_stage
        self.last_stage = last_stage
        self.init_std = init_std
        self.num_layers = num_layers

        bert_config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * mlp_ratio,
            max_position_embeddings=max_sequence_length,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
        )

        self.pooler = heads.Pooler(hidden_size)
        self.token_type_embeddings = nn.Embedding(2, hidden_size)
        self.token_type_embeddings.apply(objectives.init_weights)
        self.text_embedding = BertEmbeddings(bert_config)
        self.vis_embedding = ViTEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embedding_dim=dim,
            dropout=dropout,
            dtype=dtype,
            init_method=init_method)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
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
            ) for i in range(depth)
        ]
        norm = col_nn.LayerNorm(normalized_shape=dim, eps=layernorm_epsilon, dtype=dtype)

        if self.last_stage:
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)

        self.layer_norm = LayerNorm(hidden_size)

        layers = []
        layers.extend(blocks)
        layers.extend([norm])
        self.layers = nn.Sequential(
            *layers
        )
        # self.layers = build_pipeline_model(self.layers, num_chunks=1, verbose=True)

    def infer(self, x, image_token_type_idx=1):
        do_mlm = "_mlm"
        if f"image_{image_token_type_idx - 1}" in x:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = "image"
        img = x[imgkey]
        text_ids = x[f"text_ids{do_mlm}"]
        text_labels = x[f"text_labels{do_mlm}"]
        image_embeds = self.vis_embedding(img)
        text_embeds = self.text_embedding(text_ids)
        co_embeds = torch.cat([text_embeds, image_embeds], dim=1)
        x = co_embeds
        x = self.layers(x)
        text_feats, image_feats = (
            x[:, : text_embeds.shape[1]],
            x[:, text_embeds.shape[1] :],
        )
        cls_feats = self.pooler(x)
        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "raw_cls_feats": x[:, 0],
            "text_labels": text_labels,
            "text_ids": text_ids,
        }
        return ret

    def forward(self, x):
        ret = dict()
        ret.update(self.compute_mlm(x))
        return ret

    def compute_mlm(self, batch):
        infer = self.infer(batch)
        mlm_logits = self.mlm_score(infer["text_feats"])
        mlm_labels = infer["text_labels"]

        mlm_loss = F.cross_entropy(
            mlm_logits.view(-1, self.vocab_size),
            mlm_labels.view(-1),
            ignore_index=-100,
        )

        ret = {
            "mlm_loss": mlm_loss,
            "mlm_logits": mlm_logits,
            "mlm_labels": mlm_labels,
            "mlm_ids": infer["text_ids"],
        }

        return ret


def get_current_device():
    '''
    Returns the index of a currently selected device (gpu/cpu).
    '''
    if torch.cuda.is_available():
        return torch.cuda.current_device()
    else:
        return 'cpu'
