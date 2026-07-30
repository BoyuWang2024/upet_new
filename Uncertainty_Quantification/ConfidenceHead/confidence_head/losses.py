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

    if force_logits.ndim != 3 or force_logits.shape[1] != 3:
        raise ValueError("force_logits must have shape [N, 3, B]")
    atom_count, _, force_bin_count = force_logits.shape
    if force_labels.shape != (atom_count, 3):
        raise ValueError("force_labels must have shape [N, 3]")

    if energy_logits.ndim != 2:
        raise ValueError("energy_logits must have shape [S, B]")
    structure_count, energy_bin_count = energy_logits.shape
    if energy_labels.shape != (structure_count,):
        raise ValueError("energy_labels must have shape [S]")

    if atom_count == 0:
        raise ValueError("force tensors must contain at least one atom")
    if structure_count == 0:
        raise ValueError("energy tensors must contain at least one structure")
    if force_bin_count == 0 or energy_bin_count == 0:
        raise ValueError("logits must contain at least one bin")
    if force_bin_count != energy_bin_count:
        raise ValueError("force and energy logits must use the same bin count")

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
