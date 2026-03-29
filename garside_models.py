from typing import Mapping, Optional

import torch
import torch.nn as nn

from garside_transformer import PERMUTATIONS_S4, PolynomialMatrixTransformer, TransformerConfig


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class BurauEmbeddingMLP(nn.Module):
    """
    Embed categorical mod-p entries with learned (degree,row,col) embeddings,
    flatten the tensor, and classify the final Garside factor.
    """

    def __init__(
        self,
        p: int,
        D: int,
        matrix_size: int = 3,
        embed_dim: int = 32,
        hidden_dim: int = 1024,
        blocks: int = 3,
        dropout: float = 0.1,
        use_aux_head: bool = True,
        use_min_degree: bool = True,
        use_garside_length: bool = False,
    ):
        super().__init__()
        self.p = int(p)
        self.D = int(D)
        self.matrix_size = int(matrix_size)
        self.use_aux_head = bool(use_aux_head)
        self.use_min_degree = bool(use_min_degree)
        self.use_garside_length = bool(use_garside_length)

        self.value_emb = nn.Embedding(self.p, embed_dim)
        self.depth_emb = nn.Embedding(self.D, embed_dim)
        self.row_emb = nn.Embedding(self.matrix_size, embed_dim)
        self.col_emb = nn.Embedding(self.matrix_size, embed_dim)

        flat_dim = self.D * self.matrix_size * self.matrix_size * embed_dim
        self.input_proj = nn.Sequential(
            nn.LayerNorm(flat_dim),
            nn.Linear(flat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if self.use_min_degree:
            self.min_degree_proj = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        if self.use_garside_length:
            self.garside_length_proj = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        self.blocks = nn.Sequential(
            *[ResidualMLPBlock(hidden_dim=hidden_dim, dropout=dropout) for _ in range(blocks)]
        )
        self.trunk_norm = nn.LayerNorm(hidden_dim)
        self.factor_head = nn.Linear(hidden_dim, len(PERMUTATIONS_S4))
        self.desc_head = nn.Linear(hidden_dim, 3)

        depth_idx = torch.arange(self.D).view(self.D, 1, 1)
        row_idx = torch.arange(self.matrix_size).view(1, self.matrix_size, 1)
        col_idx = torch.arange(self.matrix_size).view(1, 1, self.matrix_size)
        self.register_buffer("depth_idx", depth_idx, persistent=False)
        self.register_buffer("row_idx", row_idx, persistent=False)
        self.register_buffer("col_idx", col_idx, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        min_degree: Optional[torch.Tensor] = None,
        garside_length: Optional[torch.Tensor] = None,
    ):
        if x.shape[1] != self.D:
            raise ValueError(f"Model D={self.D}, got input depth {x.shape[1]}")
        if tuple(x.shape[2:]) != (self.matrix_size, self.matrix_size):
            raise ValueError(
                f"Model matrix_size={self.matrix_size}, got matrix shape {tuple(x.shape[2:])}"
            )

        values = self.value_emb(x)
        positions = (
            self.depth_emb(self.depth_idx)
            + self.row_emb(self.row_idx)
            + self.col_emb(self.col_idx)
        )
        hidden = values + positions.unsqueeze(0)
        hidden = hidden.flatten(start_dim=1)
        hidden = self.input_proj(hidden)
        if self.use_min_degree:
            if min_degree is None:
                raise ValueError("Model expects min_degree input")
            min_degree = min_degree.to(dtype=torch.float32, device=x.device).view(-1, 1)
            hidden = hidden + self.min_degree_proj(min_degree)
        if self.use_garside_length:
            if garside_length is None:
                raise ValueError("Model expects garside_length input")
            garside_length = garside_length.to(dtype=torch.float32, device=x.device).view(-1, 1)
            hidden = hidden + self.garside_length_proj(garside_length)
        hidden = self.blocks(hidden)
        hidden = self.trunk_norm(hidden)

        factor_logits = self.factor_head(hidden)
        desc_logits = self.desc_head(hidden) if self.use_aux_head else None
        return factor_logits, desc_logits


def build_model_from_config(config: Optional[Mapping], p: int, D: int, matrix_size: int = 3) -> nn.Module:
    config = dict(config or {})
    model_type = config.get("model_type", "mlp")
    use_aux_head = config.get("use_aux_head")
    if use_aux_head is None:
        use_aux_head = config.get("task", "multitask") != "final_factor"
    use_min_degree = bool(config.get("use_min_degree", False))
    use_garside_length = bool(config.get("use_garside_length", False))
    matrix_size = int(config.get("matrix_size", matrix_size))

    if model_type == "mlp":
        return BurauEmbeddingMLP(
            p=p,
            D=D,
            matrix_size=matrix_size,
            embed_dim=int(config.get("embed_dim", 32)),
            hidden_dim=int(config.get("hidden_dim", 1024)),
            blocks=int(config.get("blocks", 3)),
            dropout=float(config.get("dropout", 0.1)),
            use_aux_head=bool(use_aux_head),
            use_min_degree=use_min_degree,
            use_garside_length=use_garside_length,
        )

    if model_type == "transformer":
        transformer_config = TransformerConfig(
            p=p,
            max_degree=D,
            matrix_size=matrix_size,
            d_model=int(config.get("d_model", 256)),
            ffn_mult=int(round(float(config.get("ffn_mult", 4.0)))),
            num_local_blocks=int(config.get("num_local_blocks", 2)),
            num_local_heads=int(config.get("num_local_heads", 4)),
            num_global_blocks=int(config.get("num_global_blocks", 6)),
            num_global_heads=int(config.get("num_global_heads", 8)),
            dropout=float(config.get("dropout", 0.1)),
            use_aux_head=bool(use_aux_head),
            use_min_degree=use_min_degree,
            use_garside_length=use_garside_length,
        )
        return PolynomialMatrixTransformer(transformer_config)

    raise ValueError(f"Unknown model_type: {model_type}")
