from timm.models.vision_transformer import Block, named_apply, get_init_weights_vit
from mmengine.model import BaseModel
from mmcls.registry import MODELS
from functools import partial
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

class MLP(nn.Module):
    def __init__(self, in_feat, out_feat, hidden_feat, hidden_layers):
        super().__init__()
        channel_list = [in_feat] + [hidden_feat] * hidden_layers + [out_feat]
        modules = []
        for i in range(hidden_layers + 1):
            in_channel = channel_list[i]
            out_channel = channel_list[i + 1]
            modules.append(nn.Linear(in_channel, out_channel))
            if i < hidden_layers:
                modules.append(nn.ReLU())
        self.mlp = nn.Sequential(*modules)

    def forward(self, x):
        return self.mlp(x)


class SeqEmbed(nn.Module):
    """ Embedding a sequense (B, S, C) into (B, num_tokens, C')
    """
    def __init__(self, dim, seq_len, embed_dim, div=1) -> None:
        super().__init__()
        assert seq_len % div == 0, f'Sequence length {seq_len} can not '\
                                  f'be divisible by {div}'
        self.num_tokens = seq_len // div
        self.div = div
        self.out = nn.Linear(dim * div, embed_dim)
    def forward(self, x):
        x = rearrange(x, 'b (s n) c -> b s (n c)', n = self.div)
        return self.out(x)


@MODELS.register_module()
class SubwayMLP(BaseModel):
    def __init__(self,
                 sample_points,
                 num_features,
                 num_classes,
                 hidden_feat=64,
                 hidden_layers=3,
                 data_preprocessor=None,
                 init_cfg=None):
                 
        super().__init__(data_preprocessor, init_cfg)
        self.sample_points = sample_points
        self.num_features = num_features
        self.num_classes = num_classes

        input_dim = sample_points * num_features
        self.classifier = MLP(input_dim, num_classes, 
                              hidden_feat=hidden_feat, hidden_layers=hidden_layers)

    def forward(self, data: torch.Tensor, label, mode='loss'):
        x = data.flatten(1)  # (B, S, C) -> (B, SxC)
        logits = self.classifier(x)
        loss = F.cross_entropy(logits, label)
        if mode == 'loss':
            return dict(loss=loss)
        elif mode == 'predict':
            return torch.argmax(logits, dim=1)
        else:
            raise ValueError(f'Invalid mode {mode}')


@MODELS.register_module()
class SubwayFormer(BaseModel):
    """ Modified from timm's Vision Transformer
    """
    def __init__(
            self,
            dim,        # input dim, num_features
            seq_len,    # raw seq length 
            div=10,     # devide factor
            embed_dim=64,
            num_classes=8,
            depth=1,
            num_heads=4,
            mlp_ratio=4.,
            qkv_bias=True,
            init_values=None,
            drop_rate=0.,       # lienar dropout
            attn_drop_rate=0.,  # attention dropout
            drop_path_rate=0.,
            weight_init=False,
            embed_layer=SeqEmbed,
            norm_layer=None,
            act_layer=None,
            block_fn=Block,
    ):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = embed_layer(dim, seq_len, embed_dim, div)
        num_tokens = self.patch_embed.num_tokens

        # calculated through SeqEmbed
        embed_len = num_tokens
        # we should use sin positional embedding
        self.pos_embed = self.sinusoidal_embedding(embed_len, embed_dim) # (1, S, C)
        self.pos_drop = nn.Identity()   # We do not dropout position info
        self.norm_pre = nn.Identity()   # We do not pre-norm

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                init_values=init_values,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer
            )
            for i in range(depth)])
        self.norm = nn.Identity()   # do this in fc norm

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim)
        self.head = nn.Linear(self.embed_dim, num_classes)

        if weight_init:
            self.init_weights(weight_init)

    def init_weights(self, mode=''):
        named_apply(get_init_weights_vit(mode, 0), self)

    @staticmethod
    def sinusoidal_embedding(n_channels, dim):
        pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                                for p in range(n_channels)]).cuda()
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return rearrange(pe, '... -> 1 ...')
        
    def _pos_embed(self, x):
        x = x + self.pos_embed
        return self.pos_drop(x)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x):
        x = x.mean(dim=1)   # global average
        x = self.fc_norm(x)
        return self.head(x)

    def forward(self, x, label, mode='loss'):
        x = self.forward_features(x)
        logits = self.forward_head(x)

        if mode == 'loss':
            loss = F.cross_entropy(logits, label)
            return dict(loss=loss)
        elif mode == 'predict':
            return torch.argmax(logits, dim=1)
