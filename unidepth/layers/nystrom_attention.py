import pdb
from functools import partial

import torch
import torch.nn as nn
from typing import Union
import torch.nn.functional as F
from einops import rearrange
from xformers.components.attention import NystromAttention

from .attention import AttentionBlock


class NystromBlock(AttentionBlock):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        expansion: int = 4,
        dropout: float = 0.0,
        cosine: bool = False,
        gated: bool = False,
        layer_scale: float = 1.0,
        context_dim: Union[int, None] = None,
    ):
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            expansion=expansion,
            dropout=dropout,
            cosine=cosine,
            gated=gated,
            layer_scale=layer_scale,
            context_dim=context_dim,
        )
        # pdb.set_trace()
        self.attention_fn = NystromAttention(
            num_landmarks=128, num_heads=num_heads, dropout=dropout
        )

    def attn(
        self,
        x: torch.Tensor,
        attn_bias: Union[torch.Tensor, None] = None,
        context: Union[torch.Tensor, None] = None,
        pos_embed: Union[torch.Tensor, None] = None,
        pos_embed_context: Union[torch.Tensor, None] = None,
        rope: Union[nn.Module, None] = None,
    ) -> torch.Tensor:
        x = self.norm_attnx(x)
        context = self.norm_attnctx(context)
        k, v = rearrange(
            self.kv(context), "b n (kv h d) -> b n h d kv", h=self.num_heads, kv=2
        ).unbind(dim=-1)
        q = rearrange(self.q(x), "b n (h d) -> b n h d", h=self.num_heads)
        if rope is not None:
            q = rope(q)
            k = rope(k)
        else:
            if pos_embed is not None:
                pos_embed = rearrange(
                    pos_embed, "b n (h d) -> b n h d", h=self.num_heads
                )
                q = q + pos_embed
            if pos_embed_context is not None:
                pos_embed_context = rearrange(
                    pos_embed_context, "b n (h d) -> b n h d", h=self.num_heads
                )
                k = k + pos_embed_context

        if self.cosine:
            q, k = map(partial(F.normalize, p=2, dim=-1), (q, k))  # cosine sim
        x = self.attention_fn(q, k, v, key_padding_mask=attn_bias)
        x = rearrange(x, "b n h d -> b n (h d)")
        x = self.out(x)
        return x

    # TODO: debug
    # def forward(
    #     self,
    #     x: torch.Tensor,
    #     attn_bias: Union[torch.Tensor, None] = None,
    #     context: Union[torch.Tensor, None] = None,
    #     pos_embed: Union[torch.Tensor, None] = None,
    #     pos_embed_context: Union[torch.Tensor, None] = None,
    #     rope: Union[nn.Module, None] = None,
    # ) -> torch.Tensor:
    #     print('[Debug] [Forward] In class NystromBlock(AttentionBlock)...')
    #     pdb.set_trace()
    #     context = x if context is None else context
    #     x = (
    #         self.ls1(
    #             self.attn(
    #                 x,
    #                 rope=rope,
    #                 attn_bias=attn_bias,
    #                 context=context,
    #                 pos_embed=pos_embed,
    #                 pos_embed_context=pos_embed_context,
    #             )
    #         )
    #         + x
    #     )
    #     x = self.ls2(self.mlp(x)) + x
    #     return x