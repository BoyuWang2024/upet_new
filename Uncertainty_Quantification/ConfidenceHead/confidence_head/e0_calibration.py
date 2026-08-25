"""Float64 numerical core for MAD r2SCAN E0 postprocessing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ase.io import iread
from torch import Tensor

from .artifacts import sha256_file


@dataclass(frozen=True)
class E0Dataset:
    """Composition and the two distinct energy references for one extxyz split."""

    structure_ids: Tensor
    atomic_numbers: Tensor
    atom_offsets: Tensor
    composition: Tensor
    target_energy_r2scan: Tensor
    atomization_energy: Tensor

    @property
    def atom_counts(self) -> Tensor:
        return self.atom_offsets[1:] - self.atom_offsets[:-1]


@dataclass(frozen=True)
class SvdSolution:
    """Full-rank minimum-norm SVD result and audit diagnostics."""

    solution: Tensor
    rank: int
    singular_values: Tensor
    condition_number: float
    residual_rmse: float
    residual_max_abs: float


def _float64_vector(value: Tensor | Sequence[float], name: str) -> Tensor:
    result = torch.as_tensor(value, dtype=torch.float64).reshape(-1)
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} must contain only finite values")
    return result


def solve_full_rank_svd(matrix: Tensor, rhs: Tensor) -> SvdSolution:
    """Solve an overdetermined full-column-rank system in float64 using SVD."""

    design = torch.as_tensor(matrix, dtype=torch.float64)
    target = _float64_vector(rhs, "SVD rhs")
    if design.ndim != 2 or design.shape[0] != target.numel() or design.shape[1] == 0:
        raise ValueError("SVD matrix and rhs shapes are invalid")
    if not bool(torch.isfinite(design).all()):
        raise ValueError("SVD matrix must contain only finite values")
    u, singular_values, vh = torch.linalg.svd(design, full_matrices=False)
    rank = int(torch.linalg.matrix_rank(design).item())
    if rank != design.shape[1]:
        raise ValueError(
            f"composition matrix is rank deficient: rank {rank} != {design.shape[1]}"
        )
    projected = u.mT @ target
    solution = vh.mT @ (projected / singular_values)
    residual = design @ solution - target
    return SvdSolution(
        solution=solution,
        rank=rank,
        singular_values=singular_values,
        condition_number=float((singular_values[0] / singular_values[-1]).item()),
        residual_rmse=float(torch.sqrt(torch.mean(residual.square())).item()),
        residual_max_abs=float(torch.max(torch.abs(residual)).item()),
    )


def fit_direct_mad_e0(
    composition: Tensor,
    target_energy_r2scan: Tensor,
    atomization_energy: Tensor,
) -> SvdSolution:
    """Recover MAD atomic references from total minus atomization energy."""

    target = _float64_vector(target_energy_r2scan, "target energy")
    atomization = _float64_vector(atomization_energy, "atomization energy")
    if target.shape != atomization.shape:
        raise ValueError("target and atomization energies must have matching shapes")
    return solve_full_rank_svd(composition, target - atomization)


def fit_model_aware(
    composition: Tensor,
    target_energy_r2scan: Tensor,
    model_energy_raw: Tensor,
) -> SvdSolution:
    """Fit unweighted total-energy residuals as the paper's E0 correction."""

    target = _float64_vector(target_energy_r2scan, "target energy")
    raw = _float64_vector(model_energy_raw, "raw model energy")
    if target.shape != raw.shape:
        raise ValueError("target and raw energies must have matching shapes")
    return solve_full_rank_svd(composition, target - raw)


def _apply_composition(
    model_energy_raw: Tensor,
    composition: Tensor,
    correction: Tensor,
) -> Tensor:
    raw = _float64_vector(model_energy_raw, "raw model energy")
    design = torch.as_tensor(composition, dtype=torch.float64)
    vector = _float64_vector(correction, "E0 correction")
    if design.ndim != 2 or design.shape != (raw.numel(), vector.numel()):
        raise ValueError("energy, composition, and E0 correction shapes mismatch")
    if not bool(torch.isfinite(design).all()):
        raise ValueError("composition must contain only finite values")
    return raw + design @ vector


def apply_direct_e0(
    model_energy_raw: Tensor,
    composition: Tensor,
    model_e0: Tensor,
    mad_e0: Tensor,
) -> Tensor:
    """Remove checkpoint E0 and add test-informed MAD E0."""

    checkpoint = _float64_vector(model_e0, "checkpoint E0")
    target = _float64_vector(mad_e0, "MAD E0")
    if checkpoint.shape != target.shape:
        raise ValueError("checkpoint and MAD E0 shapes mismatch")
    return _apply_composition(
        model_energy_raw,
        composition,
        target - checkpoint,
    )


def apply_model_aware_e0(
    model_energy_raw: Tensor,
    composition: Tensor,
    delta_e0: Tensor,
) -> Tensor:
    """Apply a validation-fitted model-aware E0 delta."""

    return _apply_composition(model_energy_raw, composition, delta_e0)


def _signed_errors_per_atom(
    prediction: Tensor,
    target: Tensor,
    atom_counts: Tensor,
) -> Tensor:
    predicted = _float64_vector(prediction, "predicted energy")
    reference = _float64_vector(target, "target energy")
    counts = torch.as_tensor(atom_counts, dtype=torch.int64).reshape(-1)
    if predicted.shape != reference.shape or predicted.shape != counts.shape:
        raise ValueError("energies and atom counts must have matching shapes")
    if not bool(torch.all(counts > 0)):
        raise ValueError("atom counts must be positive")
    return (predicted - reference) / counts.to(torch.float64)


def energy_observed_errors(
    prediction: Tensor,
    target: Tensor,
    atom_counts: Tensor,
) -> Tensor:
    """Return absolute total-energy residual divided by structure atom count."""

    return torch.abs(_signed_errors_per_atom(prediction, target, atom_counts))


def energy_metrics(
    prediction: Tensor,
    target: Tensor,
    atom_counts: Tensor,
) -> dict[str, int | float]:
    """Return structure-level energy metrics in meV/atom."""

    signed = _signed_errors_per_atom(prediction, target, atom_counts)
    absolute = torch.abs(signed)
    return {
        "mae_mev_per_atom": float(1000.0 * absolute.mean().item()),
        "rmse_mev_per_atom": float(
            1000.0 * torch.sqrt(torch.mean(signed.square())).item()
        ),
        "mean_signed_error_mev_per_atom": float(1000.0 * signed.mean().item()),
        "p95_absolute_error_mev_per_atom": float(
            1000.0 * torch.quantile(absolute, 0.95).item()
        ),
        "structures": int(signed.numel()),
    }


def _scalar_from_mapping(
    mapping: Mapping[str, Any] | None,
    key: str,
    *,
    structure_index: int,
) -> float | None:
    if mapping is None or key not in mapping:
        return None
    array = np.asarray(mapping[key], dtype=np.float64)
    if array.size != 1:
        raise ValueError(f"structure {structure_index}: {key} must be scalar")
    value = float(array.reshape(-1)[0])
    if not np.isfinite(value):
        raise ValueError(f"structure {structure_index}: {key} must be finite")
    return value


def _absolute_energy(atoms: Any, structure_index: int) -> float:
    calculator = getattr(atoms, "calc", None)
    results = getattr(calculator, "results", None)
    value = _scalar_from_mapping(
        results if isinstance(results, Mapping) else None,
        "energy",
        structure_index=structure_index,
    )
    if value is None:
        value = _scalar_from_mapping(
            atoms.info,
            "energy",
            structure_index=structure_index,
        )
    if value is None:
        raise ValueError(f"structure {structure_index}: absolute energy is missing")
    return value


def _atomization_energy(atoms: Any, structure_index: int) -> float:
    value = _scalar_from_mapping(
        atoms.info,
        "atomization_energy",
        structure_index=structure_index,
    )
    if value is None:
        raise ValueError(f"structure {structure_index}: atomization_energy is missing")
    return value


def load_e0_dataset(
    path: Path,
    *,
    expected_sha256: str,
    atomic_types: Sequence[int],
) -> E0Dataset:
    """Read a filtered extxyz once while binding its bytes and energy semantics."""

    source = Path(path)
    digest_before = sha256_file(source)
    if digest_before != expected_sha256:
        raise ValueError(f"dataset SHA mismatch: {digest_before} != {expected_sha256}")
    types = tuple(int(value) for value in atomic_types)
    if not types or len(set(types)) != len(types) or tuple(sorted(types)) != types:
        raise ValueError("atomic_types must be unique and sorted")
    type_to_column = {atomic_type: column for column, atomic_type in enumerate(types)}
    structure_ids: list[int] = []
    atomic_number_parts: list[Tensor] = []
    offsets = [0]
    compositions: list[Tensor] = []
    absolute: list[float] = []
    atomization: list[float] = []
    for index, atoms in enumerate(iread(str(source), index=":", format="extxyz")):
        numbers = torch.as_tensor(np.asarray(atoms.numbers), dtype=torch.int64)
        if numbers.numel() == 0:
            raise ValueError(f"structure {index}: contains no atoms")
        row = torch.zeros(len(types), dtype=torch.float64)
        for value in numbers.tolist():
            if value not in type_to_column:
                raise ValueError(f"structure {index}: unsupported atomic type {value}")
            row[type_to_column[value]] += 1.0
        structure_ids.append(index)
        atomic_number_parts.append(numbers)
        offsets.append(offsets[-1] + len(numbers))
        compositions.append(row)
        absolute.append(_absolute_energy(atoms, index))
        atomization.append(_atomization_energy(atoms, index))
    if not structure_ids:
        raise ValueError("E0 dataset must contain at least one structure")
    digest_after = sha256_file(source)
    if digest_after != digest_before:
        raise ValueError("dataset SHA changed while parsing")
    return E0Dataset(
        structure_ids=torch.tensor(structure_ids, dtype=torch.int64),
        atomic_numbers=torch.cat(atomic_number_parts),
        atom_offsets=torch.tensor(offsets, dtype=torch.int64),
        composition=torch.stack(compositions),
        target_energy_r2scan=torch.tensor(absolute, dtype=torch.float64),
        atomization_energy=torch.tensor(atomization, dtype=torch.float64),
    )


def extract_model_e0(model: Any) -> tuple[tuple[int, ...], Tensor]:
    """Extract scalar energy composition weights ordered by atomic number."""

    candidates: list[Any] = []
    for additive in getattr(model, "additive_models", ()):
        if not hasattr(additive, "sync_tensor_maps"):
            continue
        additive.sync_tensor_maps()
        weights = getattr(getattr(additive, "model", None), "weights", None)
        if isinstance(weights, Mapping) and "energy" in weights:
            candidates.append(additive)
    if len(candidates) != 1:
        raise ValueError("checkpoint must contain exactly one energy composition model")
    composition = candidates[0]
    atomic_types = tuple(sorted(int(value) for value in composition.atomic_types))
    weights = composition.model.weights["energy"]
    if len(weights) != 1:
        raise ValueError("energy composition weights must contain one scalar block")
    block = weights.block(0)
    names = tuple(str(name) for name in block.samples.names)
    if names != ("center_type",):
        raise ValueError("energy composition samples must be center_type labels")
    values = torch.as_tensor(block.values, dtype=torch.float64)
    if values.ndim != 2 or values.shape[1] != 1:
        raise ValueError("energy composition weights must be scalar")
    center_types = torch.as_tensor(block.samples.values).reshape(-1).tolist()
    if len(center_types) != values.shape[0] or len(set(center_types)) != len(
        center_types
    ):
        raise ValueError("energy composition center_type labels are invalid")
    by_type = {
        int(atomic_type): float(value)
        for atomic_type, value in zip(
            center_types,
            values.reshape(-1).tolist(),
            strict=True,
        )
    }
    if set(by_type) != set(atomic_types):
        raise ValueError("energy composition weights do not cover atomic_types")
    result = torch.tensor(
        [by_type[value] for value in atomic_types], dtype=torch.float64
    )
    if not bool(torch.isfinite(result).all()):
        raise ValueError("energy composition weights must be finite")
    return atomic_types, result
