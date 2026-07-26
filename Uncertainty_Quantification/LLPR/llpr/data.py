"""Deterministic extxyz loading and isolated metatomic system construction."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ase import Atoms
from ase.io import iread

from .artifacts import sha256_file


ENERGY_LABEL_CANDIDATES = (
    ("info", "energy"),
    ("info", "total_energy"),
    ("info", "Energy"),
    ("info", "ENERGY"),
    ("calc.results", "energy"),
    ("calc.results", "total_energy"),
)
FORCE_LABEL_CANDIDATES = (
    ("arrays", "non_conservative_forces"),
    ("arrays", "nonconservative_forces"),
    ("arrays", "forces"),
    ("arrays", "force"),
    ("calc.results", "non_conservative_forces"),
    ("calc.results", "nonconservative_forces"),
    ("calc.results", "forces"),
    ("calc.results", "force"),
)


@dataclass(frozen=True)
class DatasetIdentity:
    path: Path
    sha256: str
    structure_count: int
    atom_count: int
    force_component_count: int


@dataclass(frozen=True)
class LLPRSample:
    index: int
    atoms: Atoms
    energy_reference_total: float
    force_reference: np.ndarray


def _container(atoms: Atoms, name: str) -> Mapping[str, Any] | None:
    if name == "info":
        return atoms.info
    if name == "arrays":
        return atoms.arrays
    if name == "calc.results":
        if atoms.calc is None:
            return None
        results = getattr(atoms.calc, "results", None)
        return results if isinstance(results, Mapping) else None
    raise AssertionError(f"unknown label container {name}")


def require_energy_label(atoms: Atoms, index: int) -> float:
    for container_name, key in ENERGY_LABEL_CANDIDATES:
        container = _container(atoms, container_name)
        if container is not None and key in container:
            array = np.asarray(container[key], dtype=np.float64)
            if array.size != 1:
                raise ValueError(
                    f"structure {index}: energy label {container_name}.{key} "
                    f"has {array.size} values"
                )
            value = float(array.reshape(-1)[0])
            if not np.isfinite(value):
                raise ValueError(f"structure {index}: energy label is not finite")
            return value
    raise ValueError(f"structure {index}: no supported energy label found")


def require_force_label(atoms: Atoms, index: int) -> np.ndarray:
    for container_name, key in FORCE_LABEL_CANDIDATES:
        container = _container(atoms, container_name)
        if container is not None and key in container:
            value = np.asarray(container[key], dtype=np.float64)
            if value.ndim == 1 and value.size % 3 == 0:
                value = value.reshape(-1, 3)
            expected = (len(atoms), 3)
            if value.shape != expected:
                raise ValueError(
                    f"structure {index}: force shape {value.shape} != {expected}"
                )
            if not np.all(np.isfinite(value)):
                raise ValueError(f"structure {index}: force label is not finite")
            return value.copy()
    raise ValueError(f"structure {index}: no supported force label found")


def iter_samples(path: Path) -> Iterator[LLPRSample]:
    """Yield every structure exactly once in extxyz file order."""
    for index, atoms in enumerate(iread(str(path), index=":", format="extxyz")):
        yield LLPRSample(
            index=index,
            atoms=atoms,
            energy_reference_total=require_energy_label(atoms, index),
            force_reference=require_force_label(atoms, index),
        )


def dataset_identity(path: Path) -> DatasetIdentity:
    structures = 0
    atoms = 0
    for sample in iter_samples(path):
        structures += 1
        atoms += len(sample.atoms)
    return DatasetIdentity(
        path=Path(path).resolve(),
        sha256=sha256_file(path),
        structure_count=structures,
        atom_count=atoms,
        force_component_count=3 * atoms,
    )


def build_system(
    sample: LLPRSample,
    model: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
):
    """Build one metatomic System; private API use is isolated here."""
    from metatomic.torch import register_autograd_neighbors
    from metatomic.torch.ase_calculator import _compute_ase_neighbors
    from metatomic.torch.systems_to_torch import systems_to_torch

    system = systems_to_torch(
        sample.atoms,
        dtype=dtype,
        device=device,
        positions_requires_grad=False,
        cell_requires_grad=False,
    )
    for options in model.requested_neighbor_lists():
        neighbors = _compute_ase_neighbors(
            sample.atoms, options, dtype=dtype, device=device
        )
        register_autograd_neighbors(system, neighbors, check_consistency=False)
        system.add_neighbor_list(options, neighbors)
    return system
