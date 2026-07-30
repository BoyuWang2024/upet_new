from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class ShiftedSoftplus(nn.Module):
    """Softplus activation shifted to equal zero at the origin."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.softplus(inputs) - math.log(2.0)


class ConfidenceHead(nn.Module):
    """Multi-layer confidence classifier."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        dropout: float,
        num_bins: int,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(ShiftedSoftplus())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, num_bins))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class ComponentConfidenceHead(nn.Module):
    """Three independent confidence classifiers for vector components."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        dropout: float,
        num_bins: int,
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [
                ConfidenceHead(input_dim, hidden_dims, dropout, num_bins)
                for _ in range(3)
            ]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.stack([head(features) for head in self.heads], dim=1)
