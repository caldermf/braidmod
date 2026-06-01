"""Small absolute-degree transformer for B_3 Burau descent prediction."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from interp.b3_data import absolute_depth_for_length


@dataclass
class B3TransformerConfig:
    length: int = 25
    d_model: int = 128
    num_layers: int = 2
    num_heads: int = 4
    ffn_mult: int = 4
    dropout: float = 0.0

    @property
    def absolute_depth(self) -> int:
        return absolute_depth_for_length(self.length)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["absolute_depth"] = self.absolute_depth
        return data


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_mult: int, dropout: float):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, *, need_weights: bool = False):
        attn_in = self.attn_norm(x)
        attn_out, attn_weights = self.attn(
            attn_in,
            attn_in,
            attn_in,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        x = x + self.attn_dropout(attn_out)
        x = x + self.ffn(self.ffn_norm(x))
        return x, attn_weights


class B3AbsoluteTransformer(nn.Module):
    """Binary classifier over absolute-degree 16-way Burau slice tokens."""

    def __init__(self, config: B3TransformerConfig):
        super().__init__()
        self.config = config
        self.absolute_depth = config.absolute_depth
        self.slice_emb = nn.Embedding(16, config.d_model)
        self.degree_emb = nn.Embedding(self.absolute_depth, config.d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.blocks = nn.ModuleList(
            [
                EncoderBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    ffn_mult=config.ffn_mult,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, 1)
        self.register_buffer("degree_idx", torch.arange(self.absolute_depth), persistent=False)
        nn.init.normal_(self.cls, mean=0.0, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        return_cache: bool = False,
        need_weights: bool = False,
    ):
        if tokens.ndim != 2 or tokens.shape[1] != self.absolute_depth:
            raise ValueError(f"Expected tokens with shape [B, {self.absolute_depth}], got {tuple(tokens.shape)}")
        batch_size = tokens.shape[0]
        hidden = self.slice_emb(tokens)
        hidden = hidden + self.degree_emb(self.degree_idx).unsqueeze(0)
        hidden = torch.cat([self.cls.expand(batch_size, -1, -1), hidden], dim=1)

        cache = {"resid_pre": [], "resid_post": [], "attn_weights": []} if return_cache else None
        for block in self.blocks:
            if return_cache:
                cache["resid_pre"].append(hidden)
            hidden, attn_weights = block(hidden, need_weights=need_weights)
            if return_cache:
                cache["resid_post"].append(hidden)
                cache["attn_weights"].append(attn_weights)

        hidden = self.final_norm(hidden)
        logits = self.head(hidden[:, 0]).squeeze(-1)
        if return_cache:
            cache["final_hidden"] = hidden
            return logits, cache
        return logits
