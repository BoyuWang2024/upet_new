from __future__ import annotations

import torch
from torch import Tensor


def force_component_error(prediction: Tensor, reference: Tensor) -> Tensor:
    if (
        prediction.shape != reference.shape
        or prediction.ndim != 2
        or prediction.shape[1] != 3
    ):
        raise ValueError("force tensors must have matching [N, 3] shapes")
    if not torch.isfinite(prediction).all() or not torch.isfinite(reference).all():
        raise ValueError("force tensors must contain only finite values")

    return torch.abs(prediction - reference)


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
