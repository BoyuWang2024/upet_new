"""Dataset-size-independent signatures for canonical prediction stores."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from .errors import HardFailure
from .prediction import load_prediction_arrays, load_target_arrays, validate_predictions


def _layout(array: np.ndarray, *, leading_axes: int) -> dict[str, object]:
    return {
        "rank": array.ndim,
        "trailing_shape": list(array.shape[leading_axes:]),
        "dtype_kind": array.dtype.kind,
    }


def prediction_schema_signature(
    prediction_root: str | Path,
    *,
    split: str,
    modes: Iterable[str],
    member_count: int,
) -> dict[str, object]:
    """Validate a store and describe its schema without binding dataset size."""

    if split not in {"val", "test"}:
        raise HardFailure("prediction split must be val or test")
    normalized_modes = tuple(modes)
    if not normalized_modes or any(
        mode not in {"raw", "ema"} for mode in normalized_modes
    ):
        raise HardFailure("prediction modes must be raw or ema")
    if isinstance(member_count, bool) or member_count < 1:
        raise HardFailure("prediction member_count must be positive")

    root = Path(prediction_root).expanduser().resolve()
    targets = load_target_arrays(root / split / "targets.npz")
    target_layout = {
        "structure_ids": _layout(targets.structure_ids, leading_axes=1),
        "num_atoms": _layout(targets.num_atoms, leading_axes=1),
        "atom_offsets": _layout(targets.atom_offsets, leading_axes=1),
        "energy": _layout(targets.energy, leading_axes=1),
        "forces": _layout(targets.forces, leading_axes=1),
    }
    if targets.stress is not None:
        target_layout["stress"] = _layout(targets.stress, leading_axes=1)
    member_layout: dict[str, dict[str, object]] | None = None
    for index in range(member_count):
        for mode in normalized_modes:
            values = load_prediction_arrays(
                root / split / "members" / f"member_{index:03d}" / f"{mode}.npz"
            )
            validate_predictions(values, targets)
            current = {
                "energy": _layout(values.energy, leading_axes=1),
                "forces": _layout(values.forces, leading_axes=1),
                "stress": _layout(values.stress, leading_axes=1),
            }
            if member_layout is None:
                member_layout = current
            elif current != member_layout:
                raise HardFailure("prediction member schema differs")
    assert member_layout is not None
    return {
        "schema": (
            "upet.bootstrap.predictions/v1"
            if targets.stress is not None
            else "upet.bootstrap.predictions/v2"
        ),
        "modes": sorted(set(normalized_modes)),
        "targets": target_layout,
        "member": member_layout,
    }
