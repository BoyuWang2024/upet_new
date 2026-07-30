from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class LossOutput:
    total: torch.Tensor
    force: torch.Tensor
    energy: torch.Tensor
    force_count: int
    energy_count: int


def confidence_loss(
    force_logits: torch.Tensor,
    force_labels: torch.Tensor,
    energy_logits: torch.Tensor,
    energy_labels: torch.Tensor,
    force_weight: float = 1.0,
    energy_weight: float = 1.5,
) -> LossOutput:
    """Compute mean force-component and structure-energy classification loss."""

    force = F.cross_entropy(
        force_logits.reshape(-1, force_logits.shape[-1]),
        force_labels.reshape(-1),
    )
    energy = F.cross_entropy(energy_logits, energy_labels)
    total = force_weight * force + energy_weight * energy
    return LossOutput(
        total=total,
        force=force,
        energy=energy,
        force_count=force_labels.numel(),
        energy_count=energy_labels.numel(),
    )
