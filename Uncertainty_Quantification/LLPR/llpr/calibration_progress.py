"""Identity-bound, structure-atomic calibration progress."""

from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from .artifacts import atomic_npz_save


def _concatenate(values: list[torch.Tensor]) -> np.ndarray:
    if not values:
        return np.empty(0, dtype=np.float64)
    return (
        torch.cat(values)
        .detach()
        .reshape(-1)
        .to(dtype=torch.float64, device="cpu")
        .numpy()
    )


def save_calibration_progress(
    path: Path,
    *,
    identity: str,
    next_structure_index: int,
    energy_residuals: list[torch.Tensor],
    force_residuals: list[torch.Tensor],
    q_values: Mapping[str, list[list[torch.Tensor]]],
) -> None:
    """Atomically checkpoint complete-structure calibration accumulators."""
    arrays: dict[str, np.ndarray] = {
        "identity": np.array(identity),
        "next_structure_index": np.array(next_structure_index, dtype=np.int64),
        "energy_residuals": _concatenate(energy_residuals),
        "force_residuals": _concatenate(force_residuals),
    }
    for target in ("energy", "force"):
        for index, values in enumerate(q_values[target]):
            arrays[f"{target}_q_{index}"] = _concatenate(values)
    atomic_npz_save(path, arrays)


def load_calibration_progress(
    path: Path,
    *,
    expected_identity: str,
    candidate_counts: Mapping[str, int],
) -> tuple[
    list[torch.Tensor],
    list[torch.Tensor],
    dict[str, list[list[torch.Tensor]]],
    int,
]:
    """Load and validate one calibration checkpoint."""
    with np.load(path, allow_pickle=False) as archive:
        identity = str(archive["identity"].item())
        if identity != expected_identity:
            raise ValueError(
                f"calibration progress identity mismatch: "
                f"{identity} != {expected_identity}"
            )
        next_index = int(archive["next_structure_index"].item())
        if next_index < 0:
            raise ValueError("calibration progress next index must be non-negative")
        energy = torch.from_numpy(archive["energy_residuals"].copy()).to(torch.float64)
        force = torch.from_numpy(archive["force_residuals"].copy()).to(torch.float64)
        if not bool(torch.isfinite(energy).all()) or not bool(
            torch.isfinite(force).all()
        ):
            raise ValueError("calibration progress residuals must be finite")
        q_values: dict[str, list[list[torch.Tensor]]] = {}
        for target, residuals in (("energy", energy), ("force", force)):
            q_values[target] = []
            for index in range(candidate_counts[target]):
                key = f"{target}_q_{index}"
                if key not in archive:
                    raise ValueError(f"calibration progress is missing {key}")
                q = torch.from_numpy(archive[key].copy()).to(torch.float64)
                if (
                    q.numel() != residuals.numel()
                    or not bool(torch.isfinite(q).all())
                    or bool((q <= 0).any())
                ):
                    raise ValueError(f"calibration progress has invalid {key}")
                q_values[target].append([q] if q.numel() else [])
    return (
        [energy] if energy.numel() else [],
        [force] if force.numel() else [],
        q_values,
        next_index,
    )
