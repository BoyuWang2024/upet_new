"""Deterministic extxyz loading for confidence-head datasets."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import iread

from .artifacts import sha256_file


ENERGY_LABEL_CANDIDATES = (
    "energy",
    "free_energy",
    "dft_energy",
    "REF_energy",
)
FORCE_LABEL_CANDIDATES = (
    "forces",
    "force",
    "dft_forces",
    "REF_forces",
)


@dataclass(frozen=True)
class ConfidenceSample:
    index: int
    atoms: Atoms
    energy_reference_total: float
    force_reference: np.ndarray


@dataclass(frozen=True)
class DatasetIdentity:
    path: Path
    sha256: str
    structure_count: int
    atom_count: int
    force_component_count: int


def _calculator_results(atoms: Atoms) -> Mapping[str, Any] | None:
    if atoms.calc is None:
        return None
    results = getattr(atoms.calc, "results", None)
    return results if isinstance(results, Mapping) else None


def require_energy_label(atoms: Atoms, index: int) -> float:
    """Return the first supported scalar, finite total-energy label."""
    calculator_results = _calculator_results(atoms)
    containers = (atoms.info, calculator_results)
    for key in ENERGY_LABEL_CANDIDATES:
        for container in containers:
            if container is None or key not in container:
                continue
            array = np.asarray(container[key], dtype=np.float64)
            if array.size != 1:
                raise ValueError(
                    f"structure {index}: energy label {key!r} must be scalar"
                )
            value = float(array.reshape(-1)[0])
            if not np.isfinite(value):
                raise ValueError(
                    f"structure {index}: energy label {key!r} must be finite"
                )
            return value
    raise ValueError(f"structure {index}: no supported energy label found")


def require_force_label(atoms: Atoms, index: int) -> np.ndarray:
    """Return the first supported finite force array with exact shape ``[N, 3]``."""
    calculator_results = _calculator_results(atoms)
    containers = (atoms.arrays, calculator_results)
    expected_shape = (len(atoms), 3)
    for key in FORCE_LABEL_CANDIDATES:
        for container in containers:
            if container is None or key not in container:
                continue
            value = np.asarray(container[key], dtype=np.float64)
            if value.shape != expected_shape:
                raise ValueError(
                    f"structure {index}: force label {key!r} shape "
                    f"{value.shape} != {expected_shape}"
                )
            if not np.all(np.isfinite(value)):
                raise ValueError(
                    f"structure {index}: force label {key!r} must be finite"
                )
            return value.copy()
    raise ValueError(f"structure {index}: no supported force label found")


def iter_samples(path: Path) -> Iterator[ConfidenceSample]:
    """Yield every structure once, preserving its order in an extxyz file."""
    for index, atoms in enumerate(iread(str(path), index=":", format="extxyz")):
        yield ConfidenceSample(
            index=index,
            atoms=atoms,
            energy_reference_total=require_energy_label(atoms, index),
            force_reference=require_force_label(atoms, index),
        )


def dataset_identity(path: Path) -> DatasetIdentity:
    """Bind dataset bytes to its structure, atom, and force-component counts."""
    structure_count = 0
    atom_count = 0
    for sample in iter_samples(path):
        structure_count += 1
        atom_count += len(sample.atoms)
    return DatasetIdentity(
        path=Path(path).resolve(),
        sha256=sha256_file(path),
        structure_count=structure_count,
        atom_count=atom_count,
        force_component_count=3 * atom_count,
    )
