from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .config import ForceTargetMode


@dataclass(frozen=True)
class LossOutput:
    total: torch.Tensor
    force: torch.Tensor | None
    energy: torch.Tensor | None
    force_count: int
    energy_count: int


def _force_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mode: ForceTargetMode,
) -> tuple[torch.Tensor, int]:
    if mode == "atom_mean":
        if logits.ndim != 2:
            raise ValueError("atom-mean force_logits must have shape [N, B]")
        atom_count, bin_count = logits.shape
        if labels.shape != (atom_count,):
            raise ValueError("atom-mean force_labels must have shape [N]")
        inputs = logits
        targets = labels
    elif mode == "component":
        if logits.ndim != 3 or logits.shape[1] != 3:
            raise ValueError("component force_logits must have shape [N, 3, B]")
        atom_count, _, bin_count = logits.shape
        if labels.shape != (atom_count, 3):
            raise ValueError("component force_labels must have shape [N, 3]")
        if bin_count == 0:
            raise ValueError("logits must contain at least one bin")
        inputs = logits.reshape(-1, bin_count)
        targets = labels.reshape(-1)
    else:
        raise ValueError(f"unsupported force target mode: {mode!r}")
    if atom_count == 0:
        raise ValueError("force tensors must contain at least one atom")
    if bin_count == 0:
        raise ValueError("logits must contain at least one bin")
    return F.cross_entropy(inputs, targets), targets.numel()


def _energy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    if logits.ndim != 2:
        raise ValueError("energy_logits must have shape [S, B]")
    structure_count, bin_count = logits.shape
    if labels.shape != (structure_count,):
        raise ValueError("energy_labels must have shape [S]")
    if structure_count == 0:
        raise ValueError("energy tensors must contain at least one structure")
    if bin_count == 0:
        raise ValueError("logits must contain at least one bin")
    return F.cross_entropy(logits, labels), labels.numel()


def confidence_loss(
    force_logits: torch.Tensor | None,
    force_labels: torch.Tensor | None,
    energy_logits: torch.Tensor | None,
    energy_labels: torch.Tensor | None,
    *,
    force_target_mode: ForceTargetMode,
    force_weight: float = 1.0,
    energy_weight: float = 1.5,
) -> LossOutput:
    """Compute a supervised weighted total over the active target branches."""
    if (force_logits is None) != (force_labels is None):
        raise ValueError("force logits and labels must be enabled together")
    if (energy_logits is None) != (energy_labels is None):
        raise ValueError("energy logits and labels must be enabled together")
    if force_logits is None and energy_logits is None:
        raise ValueError("at least one confidence target must be active")
    if force_weight < 0 or energy_weight < 0:
        raise ValueError("loss weights must be nonnegative")

    force: torch.Tensor | None = None
    energy: torch.Tensor | None = None
    force_count = 0
    energy_count = 0
    total: torch.Tensor | None = None
    if force_logits is not None and force_labels is not None:
        force, force_count = _force_loss(
            force_logits,
            force_labels,
            force_target_mode,
        )
        total = force_weight * force
    if energy_logits is not None and energy_labels is not None:
        energy, energy_count = _energy_loss(energy_logits, energy_labels)
        weighted = energy_weight * energy
        total = weighted if total is None else total + weighted
    if total is None:
        raise RuntimeError("active loss did not produce a total")
    return LossOutput(
        total=total,
        force=force,
        energy=energy,
        force_count=force_count,
        energy_count=energy_count,
    )
