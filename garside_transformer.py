from dataclasses import dataclass
from itertools import permutations
from typing import Optional

import torch
import torch.nn as nn


PERMUTATIONS_S4 = list(permutations(range(4)))


def infer_degree_mask(x: torch.Tensor) -> torch.Tensor:
    """
    Infer a contiguous valid-degree mask from a zero-padded Burau tensor.

    The dataset stores projectively normalized coefficients in degrees 0..width-1
    and pads the remainder of the fixed checkpoint depth with zeros. Internal zero
    slices inside the occupied support remain valid, so the mask is the full prefix
    up to the last occupied degree, not just the set of nonzero slices.
    """
    if x.ndim != 4:
        raise ValueError(f"Expected x with shape [B, D, M, M], got {tuple(x.shape)}")

    degree_has_support = x.ne(0).any(dim=(-1, -2))
    any_support = degree_has_support.any(dim=1)
    last_valid = x.shape[1] - 1 - degree_has_support.flip(dims=[1]).to(torch.int64).argmax(dim=1)
    positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
    mask = positions <= last_valid.unsqueeze(1)
    return mask & any_support.unsqueeze(1)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_mult: int, dropout: float):
        super().__init__()
        hidden_dim = int(ffn_mult) * int(d_model)
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
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_in = self.attn_norm(x)
        attn_out, _ = self.attn(
            attn_in,
            attn_in,
            attn_in,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.attn_dropout(attn_out)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class PolynomialMatrixEmbedder(nn.Module):
    def __init__(
        self,
        p: int,
        max_degree: int,
        matrix_size: int,
        d_model: int,
        dropout: float,
    ):
        super().__init__()
        self.p = int(p)
        self.max_degree = int(max_degree)
        self.matrix_size = int(matrix_size)
        self.d_model = int(d_model)

        self.value_emb = nn.Embedding(self.p, self.d_model)
        self.row_emb = nn.Embedding(self.matrix_size, self.d_model)
        self.col_emb = nn.Embedding(self.matrix_size, self.d_model)
        self.degree_emb_local = nn.Embedding(self.max_degree, self.d_model)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("degree_idx", torch.arange(self.max_degree), persistent=False)
        self.register_buffer("row_idx", torch.arange(self.matrix_size), persistent=False)
        self.register_buffer("col_idx", torch.arange(self.matrix_size), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected x with shape [B, D, M, M], got {tuple(x.shape)}")
        if x.shape[1] > self.max_degree:
            raise ValueError(f"Embedder max_degree={self.max_degree}, got depth {x.shape[1]}")
        if tuple(x.shape[2:]) != (self.matrix_size, self.matrix_size):
            raise ValueError(
                f"Embedder matrix_size={self.matrix_size}, got matrix shape {tuple(x.shape[2:])}"
            )

        batch_size, depth = x.shape[:2]
        values = self.value_emb(x)
        row = self.row_emb(self.row_idx).view(1, 1, self.matrix_size, 1, self.d_model)
        col = self.col_emb(self.col_idx).view(1, 1, 1, self.matrix_size, self.d_model)
        degree = self.degree_emb_local(self.degree_idx[:depth]).view(1, depth, 1, 1, self.d_model)
        tokens = values + row + col + degree
        tokens = tokens.view(batch_size, depth, self.matrix_size * self.matrix_size, self.d_model)
        return self.dropout(tokens)


class LocalMatrixEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        ffn_mult: int,
        num_heads: int,
        num_blocks: int,
        dropout: float,
    ):
        super().__init__()
        self.local_cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    ffn_mult=ffn_mult,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.output_norm = nn.LayerNorm(d_model)
        nn.init.normal_(self.local_cls, mean=0.0, std=0.02)

    def forward(
        self,
        local_tokens: torch.Tensor,
        degree_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if local_tokens.ndim != 4:
            raise ValueError(
                f"Expected local_tokens with shape [B, D, M*M, d_model], got {tuple(local_tokens.shape)}"
            )

        batch_size, depth = local_tokens.shape[:2]
        cls = self.local_cls.expand(batch_size, depth, -1, -1)
        hidden = torch.cat([cls, local_tokens], dim=2)
        hidden = hidden.reshape(batch_size * depth, hidden.shape[2], hidden.shape[3])

        for block in self.blocks:
            hidden = block(hidden)

        hidden = self.output_norm(hidden)
        summaries = hidden[:, 0, :].reshape(batch_size, depth, hidden.shape[-1])
        if degree_mask is not None:
            summaries = summaries * degree_mask.unsqueeze(-1).to(dtype=summaries.dtype)
        return summaries


class GlobalPolynomialEncoder(nn.Module):
    def __init__(
        self,
        max_degree: int,
        d_model: int,
        ffn_mult: int,
        num_heads: int,
        num_blocks: int,
        dropout: float,
    ):
        super().__init__()
        self.max_degree = int(max_degree)
        self.d_model = int(d_model)
        self.global_cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.degree_emb_global = nn.Embedding(self.max_degree, d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    ffn_mult=ffn_mult,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.register_buffer("degree_idx", torch.arange(self.max_degree), persistent=False)
        nn.init.normal_(self.global_cls, mean=0.0, std=0.02)

    def forward(
        self,
        degree_tokens: torch.Tensor,
        degree_mask: Optional[torch.Tensor] = None,
        cls_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if degree_tokens.ndim != 3:
            raise ValueError(
                f"Expected degree_tokens with shape [B, D, d_model], got {tuple(degree_tokens.shape)}"
            )
        if degree_tokens.shape[1] > self.max_degree:
            raise ValueError(f"Global encoder max_degree={self.max_degree}, got depth {degree_tokens.shape[1]}")

        batch_size, depth = degree_tokens.shape[:2]
        if degree_mask is None:
            degree_mask = torch.ones(batch_size, depth, dtype=torch.bool, device=degree_tokens.device)
        elif degree_mask.shape != (batch_size, depth):
            raise ValueError(
                f"degree_mask must have shape {(batch_size, depth)}, got {tuple(degree_mask.shape)}"
            )

        degree_positions = self.degree_emb_global(self.degree_idx[:depth]).unsqueeze(0)
        hidden = degree_tokens + degree_positions
        hidden = hidden * degree_mask.unsqueeze(-1).to(dtype=hidden.dtype)

        cls = self.global_cls.expand(batch_size, -1, -1)
        if cls_bias is not None:
            if cls_bias.shape != (batch_size, self.d_model):
                raise ValueError(
                    f"cls_bias must have shape {(batch_size, self.d_model)}, got {tuple(cls_bias.shape)}"
                )
            cls = cls + cls_bias.unsqueeze(1)

        hidden = torch.cat([cls, hidden], dim=1)
        key_padding_mask = torch.cat(
            [
                torch.zeros(batch_size, 1, dtype=torch.bool, device=degree_tokens.device),
                ~degree_mask,
            ],
            dim=1,
        )

        for block in self.blocks:
            hidden = block(hidden, key_padding_mask=key_padding_mask)

        hidden = self.output_norm(hidden)
        return hidden[:, 0, :]


@dataclass
class TransformerConfig:
    p: int
    max_degree: int
    matrix_size: int = 3
    d_model: int = 256
    ffn_mult: int = 4
    num_local_blocks: int = 2
    num_local_heads: int = 4
    num_global_blocks: int = 6
    num_global_heads: int = 8
    dropout: float = 0.1
    use_aux_head: bool = True
    use_min_degree: bool = True
    use_garside_length: bool = False


class PolynomialMatrixTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.p = int(config.p)
        self.D = int(config.max_degree)
        self.matrix_size = int(config.matrix_size)
        self.use_aux_head = bool(config.use_aux_head)
        self.use_min_degree = bool(config.use_min_degree)
        self.use_garside_length = bool(config.use_garside_length)

        self.embedder = PolynomialMatrixEmbedder(
            p=config.p,
            max_degree=config.max_degree,
            matrix_size=config.matrix_size,
            d_model=config.d_model,
            dropout=config.dropout,
        )
        self.local_encoder = LocalMatrixEncoder(
            d_model=config.d_model,
            ffn_mult=config.ffn_mult,
            num_heads=config.num_local_heads,
            num_blocks=config.num_local_blocks,
            dropout=config.dropout,
        )
        self.global_encoder = GlobalPolynomialEncoder(
            max_degree=config.max_degree,
            d_model=config.d_model,
            ffn_mult=config.ffn_mult,
            num_heads=config.num_global_heads,
            num_blocks=config.num_global_blocks,
            dropout=config.dropout,
        )
        if self.use_min_degree:
            self.min_degree_proj = nn.Sequential(
                nn.Linear(1, config.d_model),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
        if self.use_garside_length:
            self.garside_length_proj = nn.Sequential(
                nn.Linear(1, config.d_model),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
        self.output_norm = nn.LayerNorm(config.d_model)
        self.factor_head = nn.Linear(config.d_model, len(PERMUTATIONS_S4))
        self.desc_head = nn.Linear(config.d_model, 3)

    def forward(
        self,
        x: torch.Tensor,
        min_degree: Optional[torch.Tensor] = None,
        garside_length: Optional[torch.Tensor] = None,
        degree_mask: Optional[torch.Tensor] = None,
    ):
        if x.shape[1] != self.D:
            raise ValueError(f"Model D={self.D}, got input depth {x.shape[1]}")
        if tuple(x.shape[2:]) != (self.matrix_size, self.matrix_size):
            raise ValueError(
                f"Model matrix_size={self.matrix_size}, got matrix shape {tuple(x.shape[2:])}"
            )

        if degree_mask is None:
            degree_mask = infer_degree_mask(x)
        else:
            degree_mask = degree_mask.to(device=x.device, dtype=torch.bool)

        local_tokens = self.embedder(x)
        degree_tokens = self.local_encoder(local_tokens, degree_mask=degree_mask)

        cls_bias = None
        if self.use_min_degree:
            if min_degree is None:
                raise ValueError("Model expects min_degree input")
            min_degree = min_degree.to(dtype=torch.float32, device=x.device).view(-1, 1)
            cls_bias = self.min_degree_proj(min_degree)
        if self.use_garside_length:
            if garside_length is None:
                raise ValueError("Model expects garside_length input")
            garside_length = garside_length.to(dtype=torch.float32, device=x.device).view(-1, 1)
            length_bias = self.garside_length_proj(garside_length)
            cls_bias = length_bias if cls_bias is None else cls_bias + length_bias

        poly_repr = self.global_encoder(
            degree_tokens,
            degree_mask=degree_mask,
            cls_bias=cls_bias,
        )
        poly_repr = self.output_norm(poly_repr)
        factor_logits = self.factor_head(poly_repr)
        desc_logits = self.desc_head(poly_repr) if self.use_aux_head else None
        return factor_logits, desc_logits
