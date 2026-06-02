"""Simple MLP baselines for B_3 Burau descent prediction."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from interp.b3_data import absolute_depth_for_length


@dataclass
class B3MLPConfig:
    length: int = 25
    hidden_dim: int = 128
    num_hidden_layers: int = 1
    dropout: float = 0.0

    @property
    def absolute_depth(self) -> int:
        return absolute_depth_for_length(self.length)

    @property
    def input_dim(self) -> int:
        return self.absolute_depth * 16

    def to_dict(self) -> dict:
        data = asdict(self)
        data["absolute_depth"] = self.absolute_depth
        data["input_dim"] = self.input_dim
        return data


class B3AbsoluteMLP(nn.Module):
    """Flattened one-hot absolute-degree slice MLP."""

    def __init__(self, config: B3MLPConfig):
        super().__init__()
        self.config = config
        self.absolute_depth = config.absolute_depth
        layers: list[nn.Module] = []
        in_dim = config.input_dim
        for _ in range(config.num_hidden_layers):
            layers.extend(
                [
                    nn.Linear(in_dim, config.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                ]
            )
            in_dim = config.hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.shape[1] != self.absolute_depth:
            raise ValueError(f"Expected tokens with shape [B, {self.absolute_depth}], got {tuple(tokens.shape)}")
        x = F.one_hot(tokens.to(torch.long), num_classes=16).flatten(start_dim=1).to(torch.float32)
        return self.net(x).squeeze(-1)
