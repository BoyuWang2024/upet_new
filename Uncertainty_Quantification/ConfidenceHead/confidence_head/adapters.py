from __future__ import annotations

import math

import torch
from torch import nn


class LocalToGlobalCumulantAdapter(nn.Module):
    """Aggregate atom-level features into structure-level cumulants."""

    def __init__(self, input_dim: int, order: int, signed_root: bool) -> None:
        super().__init__()
        if not 1 <= order <= 8:
            raise ValueError("cumulant order must be between 1 and 8")

        self.input_dim = input_dim
        self.order = order
        self.signed_root = signed_root

    def forward(
        self,
        features: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(f"features must have shape [N, {self.input_dim}]")
        self._validate_offsets(offsets, features.shape[0])

        structures = []
        for start_tensor, stop_tensor in zip(offsets[:-1], offsets[1:], strict=False):
            start = int(start_tensor.item())
            stop = int(stop_tensor.item())
            values = features[start:stop]
            raw = [
                values.pow(degree).mean(dim=0) for degree in range(1, self.order + 1)
            ]
            cumulants: list[torch.Tensor] = []
            output_cumulants: list[torch.Tensor] = []
            for degree in range(1, self.order + 1):
                cumulant = raw[degree - 1]
                for lower_degree in range(1, degree):
                    coefficient = math.comb(degree - 1, lower_degree - 1)
                    cumulant = cumulant - (
                        coefficient
                        * cumulants[lower_degree - 1]
                        * raw[degree - lower_degree - 1]
                    )
                cumulants.append(cumulant)
                if degree > 1 and self.signed_root:
                    cumulant = torch.sign(cumulant) * torch.abs(cumulant).pow(
                        1.0 / degree
                    )
                output_cumulants.append(cumulant)
            structures.append(torch.cat(output_cumulants, dim=0))

        return torch.stack(structures, dim=0)

    @staticmethod
    def _validate_offsets(offsets: torch.Tensor, atom_count: int) -> None:
        integer_dtypes = {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
        if offsets.ndim != 1:
            raise ValueError("offsets must be one-dimensional")
        if offsets.dtype not in integer_dtypes:
            raise ValueError("offsets must have an integer dtype")
        if offsets.numel() < 2:
            raise ValueError("offsets must contain at least [0, N]")
        if int(offsets[0].item()) != 0:
            raise ValueError("offsets must start at zero")
        if int(offsets[-1].item()) != atom_count:
            raise ValueError("offsets must end at the atom count")
        if bool(torch.any(offsets[1:] <= offsets[:-1]).item()):
            raise ValueError("offsets must be strictly increasing")
