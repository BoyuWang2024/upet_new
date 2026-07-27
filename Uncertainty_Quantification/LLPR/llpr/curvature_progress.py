"""Strict validation for resumable curvature progress."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ValidatedCurvatureProgress:
    energy: np.ndarray
    force: np.ndarray
    structure_count: int
    atom_count: int
    force_component_count: int
    next_structure_index: int


def _nonnegative_integer(arrays: dict[str, np.ndarray], name: str) -> int:
    value = np.asarray(arrays[name])
    if (
        value.shape != ()
        or not np.issubdtype(value.dtype, np.integer)
        or np.issubdtype(value.dtype, np.bool_)
    ):
        raise ValueError(f"curvature progress {name} must be an integer scalar")
    result = int(value.item())
    if result < 0:
        raise ValueError(f"curvature progress {name} must be non-negative")
    return result


def load_validated_curvature_progress(
    path: Path,
    *,
    expected_identity: str,
    expected_energy_dimension: int,
    expected_force_dimension: int,
    expected_structure_count: int,
) -> ValidatedCurvatureProgress:
    """Load curvature progress only after all recovery invariants hold."""
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "identity",
            "energy",
            "force",
            "structure_count",
            "atom_count",
            "force_component_count",
            "next_structure_index",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"curvature progress fields are missing: {missing}")
        arrays = {name: archive[name].copy() for name in required}

    identity_value = np.asarray(arrays["identity"])
    if identity_value.shape != ():
        raise ValueError("curvature progress identity must be a scalar")
    identity = str(identity_value.item())
    if identity != expected_identity:
        raise ValueError(
            f"curvature identity mismatch: {identity} != {expected_identity}"
        )

    energy = np.asarray(arrays["energy"], dtype=np.float64)
    force = np.asarray(arrays["force"], dtype=np.float64)
    expected_energy_shape = (expected_energy_dimension, expected_energy_dimension)
    expected_force_shape = (expected_force_dimension, expected_force_dimension)
    if energy.shape != expected_energy_shape:
        raise ValueError(
            f"curvature progress energy matrix shape {energy.shape} "
            f"!= {expected_energy_shape}"
        )
    if force.shape != expected_force_shape:
        raise ValueError(
            f"curvature progress force matrix shape {force.shape} "
            f"!= {expected_force_shape}"
        )
    if not np.all(np.isfinite(energy)) or not np.all(np.isfinite(force)):
        raise ValueError("curvature progress matrices must be finite")
    if not np.allclose(energy, energy.T, rtol=0, atol=1.0e-12):
        raise ValueError("curvature progress energy matrix must be symmetric")
    if not np.allclose(force, force.T, rtol=0, atol=1.0e-12):
        raise ValueError("curvature progress force matrix must be symmetric")

    structure_count = _nonnegative_integer(arrays, "structure_count")
    atom_count = _nonnegative_integer(arrays, "atom_count")
    force_component_count = _nonnegative_integer(arrays, "force_component_count")
    next_index = _nonnegative_integer(arrays, "next_structure_index")
    if next_index != structure_count:
        raise ValueError("curvature progress next index does not match structure count")
    if next_index > expected_structure_count:
        raise ValueError("curvature progress next index exceeds build size")
    if force_component_count != 3 * atom_count:
        raise ValueError(
            "curvature progress force-component count does not equal 3 * atom count"
        )

    return ValidatedCurvatureProgress(
        energy=energy,
        force=force,
        structure_count=structure_count,
        atom_count=atom_count,
        force_component_count=force_component_count,
        next_structure_index=next_index,
    )
