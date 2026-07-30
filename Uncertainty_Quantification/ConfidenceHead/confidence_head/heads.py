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
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if num_bins <= 0:
            raise ValueError("num_bins must be positive")
        if any(hidden_dim <= 0 for hidden_dim in hidden_dims):
            raise ValueError("all hidden_dims must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in the interval [0, 1)")

        self.input_dim = input_dim
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
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(f"features must have shape [N, {self.input_dim}]")
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
