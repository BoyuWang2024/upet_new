from __future__ import annotations

import math

import torch
from torch import nn


class LocalToGlobalCumulantAdapter(nn.Module):
    """Vectorize atom-level features into per-structure cumulants."""

    def __init__(self, input_dim: int, order: int, signed_root: bool) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if not 1 <= order <= 8:
            raise ValueError("cumulant order must be between 1 and 8")
        self.input_dim = input_dim
        self.order = order
        self.signed_root = signed_root

    def forward(
        self,
        features: torch.Tensor,
        atom_counts: torch.Tensor,
    ) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(f"features must have shape [N, {self.input_dim}]")
        self._validate_atom_counts(atom_counts, features.shape[0])
        if atom_counts.device != features.device:
            raise ValueError("atom_counts and features must be on the same device")

        structure_count = atom_counts.shape[0]
        structure_ids = torch.repeat_interleave(
            torch.arange(structure_count, device=features.device),
            atom_counts,
        )
        denominator = atom_counts.to(dtype=features.dtype).unsqueeze(1)
        raw: list[torch.Tensor] = []
        for degree in range(1, self.order + 1):
            moment = features.new_zeros((structure_count, self.input_dim))
            moment.index_add_(0, structure_ids, features.pow(degree))
            raw.append(moment / denominator)

        cumulants: list[torch.Tensor] = []
        outputs: list[torch.Tensor] = []
        for degree in range(1, self.order + 1):
            cumulant = raw[degree - 1]
            for lower_degree in range(1, degree):
                cumulant = cumulant - (
                    math.comb(degree - 1, lower_degree - 1)
                    * cumulants[lower_degree - 1]
                    * raw[degree - lower_degree - 1]
                )
            cumulants.append(cumulant)
            if degree > 1 and self.signed_root:
                magnitude = torch.abs(cumulant).clamp_min(
                    torch.finfo(cumulant.dtype).tiny
                )
                cumulant = torch.sign(cumulant) * magnitude.pow(1 / degree)
            outputs.append(cumulant)
        return torch.cat(outputs, dim=1)

    @staticmethod
    def _validate_atom_counts(atom_counts: torch.Tensor, atom_count: int) -> None:
        integer_dtypes = {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
        if atom_counts.ndim != 1:
            raise ValueError("atom_counts must be one-dimensional")
        if atom_counts.dtype not in integer_dtypes:
            raise ValueError("atom_counts must have an integer dtype")
        if atom_counts.numel() == 0:
            raise ValueError("atom_counts must contain at least one structure")
        if bool(torch.any(atom_counts <= 0)):
            raise ValueError("atom_counts must be strictly positive")
        if bool(torch.sum(atom_counts) != atom_count):
            raise ValueError("atom_counts must sum to the feature atom count")
