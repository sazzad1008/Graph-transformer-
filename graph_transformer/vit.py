"""Vision Transformer components."""

from __future__ import annotations

from typing import Any

import math

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
from torch import Tensor, nn


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 224):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange("b e h w -> b (h w) e"),
        )
        self.class_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        b = x.size(0)
        x = self.projection(x)
        cls_tokens = repeat(self.class_token, "() n e -> b n e", b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.positions
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0) -> None:
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.d_k = emb_size // num_heads
        self.inv_sqrt_d_k = 1.0 / math.sqrt(self.d_k)
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum("b h q d, b h k d -> b h q k", queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy = energy.masked_fill(~mask, fill_value)
        att = F.softmax(energy * self.inv_sqrt_d_k, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum("b h a l, b h l v -> b h a v", att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.0) -> None:
        super().__init__()
        self.linear1 = nn.Linear(emb_size, expansion * emb_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(drop_p)
        self.linear2 = nn.Linear(expansion * emb_size, emb_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        emb_size: int = 768,
        drop_p: float = 0.0,
        forward_expansion: int = 4,
        forward_drop_p: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.attention = MultiHeadAttention(emb_size, **kwargs)
        self.dropout1 = nn.Dropout(drop_p)

        self.norm2 = nn.LayerNorm(emb_size)
        self.feedforward = FeedForward(emb_size, expansion=forward_expansion, drop_p=forward_drop_p)
        self.dropout2 = nn.Dropout(drop_p)

    def forward(self, x: Tensor) -> Tensor:
        residual1 = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.dropout1(x)
        x = x + residual1

        residual2 = x
        x = self.norm2(x)
        x = self.feedforward(x)
        x = self.dropout2(x)
        x = x + residual2

        return x


class Encoder(nn.Module):
    def __init__(self, depth: int = 12, **kwargs: Any) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([TransformerEncoderBlock(**kwargs) for _ in range(depth)])

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class ClassificationHead(nn.Module):
    def __init__(self, emb_size: int = 768, n_classes: int = 6) -> None:
        super().__init__()
        self.reduce = Reduce("b n e -> b e", reduction="mean")
        self.layer_norm = nn.LayerNorm(emb_size)
        self.linear = nn.Linear(emb_size, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.reduce(x)
        x = self.layer_norm(x)
        x = self.linear(x)
        return x


class ViT(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = 16,
        emb_size: int = 768,
        img_size: int = 224,
        depth: int = 12,
        n_classes: int = 6,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        self.encoder = Encoder(depth, emb_size=emb_size, **kwargs)
        self.classification = ClassificationHead(emb_size, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embedding(x)
        x = self.encoder(x)
        x = self.classification(x)
        return x
