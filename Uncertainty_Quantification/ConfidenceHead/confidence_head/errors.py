from __future__ import annotations

import torch
from torch import Tensor

from .config import ForceTargetMode


def _validate_force_tensors(prediction: Tensor, reference: Tensor) -> None:
    if (
        prediction.shape != reference.shape
        or prediction.ndim != 2
        or prediction.shape[1] != 3
    ):
        raise ValueError("force tensors must have matching [N, 3] shapes")
    if not torch.isfinite(prediction).all() or not torch.isfinite(reference).all():
        raise ValueError("force tensors must contain only finite values")


def force_error(
    prediction: Tensor,
    reference: Tensor,
    target_mode: ForceTargetMode,
) -> Tensor:
    _validate_force_tensors(prediction, reference)
    component_errors = torch.abs(prediction - reference)
    if target_mode == "component":
        return component_errors
    if target_mode == "atom_mean":
        return component_errors.mean(dim=-1)
    raise ValueError(f"unsupported force target mode: {target_mode!r}")


def force_error_definition(target_mode: ForceTargetMode) -> str:
    if target_mode == "component":
        return "abs_cartesian_component_v1"
    if target_mode == "atom_mean":
        return "abs_cartesian_component_mean_v1"
    raise ValueError(f"unsupported force target mode: {target_mode!r}")


def force_component_error(prediction: Tensor, reference: Tensor) -> Tensor:
    return force_error(prediction, reference, "component")


def energy_per_atom_error(
    prediction: Tensor,
    reference: Tensor,
    atom_counts: Tensor,
) -> Tensor:
    prediction = prediction.reshape(-1)
    reference = reference.reshape(-1)
    atom_counts = atom_counts.reshape(-1)

    if prediction.shape != reference.shape or prediction.shape != atom_counts.shape:
        raise ValueError("energy tensors and atom counts must have matching shapes")
    if not torch.all(atom_counts > 0):
        raise ValueError("atom counts must be positive")
    if not torch.isfinite(prediction).all() or not torch.isfinite(reference).all():
        raise ValueError("energy tensors must contain only finite values")

    return torch.abs(prediction - reference) / atom_counts.to(
        device=prediction.device,
        dtype=prediction.dtype,
    )
