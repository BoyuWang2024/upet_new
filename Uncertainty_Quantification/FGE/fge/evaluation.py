"""Pure evaluation of canonical FGE prediction payloads."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .artifacts import (
    ExperimentLayout,
    atomic_torch_save,
    atomic_write_json,
    sibling_staging,
)
from .errors import HardFailure
from .uncertainty import (
    FORMULA_VERSION,
    population_std,
    reduce_force_by_structure,
    scalar_gmd,
    vector_gmd,
)


_REQUIRED_PAYLOAD_FIELDS = frozenset(
    {
        "energy_prediction",
        "forces_prediction",
        "stress_prediction",
        "energy_reference",
        "forces_reference",
        "stress_reference",
        "n_atoms",
        "structure_offsets",
        "member_ids",
    }
)


@dataclass(frozen=True)
class EvaluationArtifacts:
    """Structured in-memory inputs for the later evaluation artifact writers."""

    ensemble: Mapping[str, Any]
    uncertainty: Mapping[str, Any]
    metrics: Mapping[str, Any]
    report_inputs: Mapping[str, Any]


def _floating_tensor(value: object, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU tensor")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must use canonical float32 dtype")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return value


def _integer_tensor(value: object, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU tensor")
    if value.dtype not in {torch.int32, torch.int64}:
        raise TypeError(f"{name} must use an integer dtype")
    return value


def global_mae(prediction: torch.Tensor, reference: torch.Tensor) -> float:
    """Return total absolute scalar error divided by the scalar element count."""
    if not isinstance(prediction, torch.Tensor) or not isinstance(
        reference, torch.Tensor
    ):
        raise TypeError("prediction and reference must be torch.Tensor instances")
    if prediction.device.type != "cpu" or reference.device.type != "cpu":
        raise ValueError("prediction and reference must be CPU tensors")
    if prediction.shape != reference.shape:
        raise ValueError("prediction and reference shapes must match")
    if prediction.numel() == 0:
        raise ValueError("prediction and reference must not be empty")
    if not prediction.is_floating_point() or not reference.is_floating_point():
        raise TypeError("prediction and reference must have floating dtypes")
    if not bool(torch.isfinite(prediction).all().item()) or not bool(
        torch.isfinite(reference).all().item()
    ):
        raise ValueError("prediction and reference must contain only finite values")
    prediction_float64 = prediction.to(dtype=torch.float64)
    reference_float64 = reference.to(dtype=torch.float64)
    return float(
        torch.abs(prediction_float64 - reference_float64).sum().item()
        / prediction.numel()
    )


def _assert_finite_derived(value: object, role: str) -> None:
    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"derived {role} must contain only finite values")
        return
    if isinstance(value, Mapping):
        for name, nested in value.items():
            _assert_finite_derived(nested, f"{role}.{name}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, nested in enumerate(value):
            _assert_finite_derived(nested, f"{role}[{index}]")
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"derived {role} must contain only finite values")


def _validate_coverages(coverages: Sequence[float]) -> tuple[float, ...]:
    if isinstance(coverages, (str, bytes)) or not isinstance(coverages, Sequence):
        raise TypeError("coverages must be a non-empty sequence")
    values: list[float] = []
    for coverage in coverages:
        if isinstance(coverage, bool) or not isinstance(coverage, (float, int)):
            raise TypeError("coverages must contain real numbers")
        value = float(coverage)
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError("coverages must lie in (0, 1]")
        values.append(value)
    if not values or any(
        left <= right for left, right in zip(values, values[1:], strict=False)
    ):
        raise ValueError("coverages must be non-empty, unique, and strictly descending")
    return tuple(values)


def _validate_payload(payload: Mapping[str, object]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    fields = set(payload)
    missing = sorted(_REQUIRED_PAYLOAD_FIELDS - fields)
    if missing:
        raise ValueError(f"payload has missing fields: {', '.join(missing)}")

    energy = _floating_tensor(payload["energy_prediction"], "energy_prediction")
    forces = _floating_tensor(payload["forces_prediction"], "forces_prediction")
    stress = _floating_tensor(payload["stress_prediction"], "stress_prediction")
    energy_reference = _floating_tensor(payload["energy_reference"], "energy_reference")
    forces_reference = _floating_tensor(payload["forces_reference"], "forces_reference")
    stress_reference = _floating_tensor(payload["stress_reference"], "stress_reference")
    n_atoms = _integer_tensor(payload["n_atoms"], "n_atoms")
    offsets = _integer_tensor(payload["structure_offsets"], "structure_offsets")

    if energy.ndim != 2:
        raise ValueError("energy_prediction must have shape [K, S]")
    member_count, structure_count = energy.shape
    if member_count < 2:
        raise ValueError("energy_prediction K must contain at least two members")
    if structure_count < 1:
        raise ValueError("energy_prediction must contain at least one structure")
    if forces.ndim != 3 or forces.shape[0] != member_count or forces.shape[2] != 3:
        raise ValueError("forces_prediction must have shape [K, A, 3] with matching K")
    atom_count = forces.shape[1]
    if stress.shape != (member_count, structure_count, 3, 3):
        raise ValueError(
            "stress_prediction must have shape [K, S, 3, 3] with matching K"
        )
    if energy_reference.shape != (structure_count,):
        raise ValueError("energy_reference must have shape [S]")
    if forces_reference.shape != (atom_count, 3):
        raise ValueError("forces_reference atom shape must match [A, 3]")
    if stress_reference.shape != (structure_count, 3, 3):
        raise ValueError("stress_reference must have shape [S, 3, 3]")
    if n_atoms.shape != (structure_count,) or not bool((n_atoms > 0).all().item()):
        raise ValueError("n_atoms must contain one positive count per structure")
    if int(n_atoms.sum().item()) != atom_count:
        raise ValueError("n_atoms sum must match the force atom count")
    if offsets.shape != (structure_count + 1,):
        raise ValueError("structure_offsets must have shape [S + 1]")
    expected_offsets = torch.cat(
        (torch.zeros(1, dtype=n_atoms.dtype), torch.cumsum(n_atoms, dim=0))
    )
    if not torch.equal(offsets.to(n_atoms.dtype), expected_offsets):
        raise ValueError("structure_offsets must exactly match cumulative n_atoms")

    member_ids = payload["member_ids"]
    if isinstance(member_ids, (str, bytes)) or not isinstance(member_ids, Sequence):
        raise ValueError("member_ids must be a sequence of unique strings")
    normalized_ids = tuple(member_ids)
    if (
        len(normalized_ids) != member_count
        or any(
            not isinstance(member_id, str) or not member_id
            for member_id in normalized_ids
        )
        or len(set(normalized_ids)) != member_count
    ):
        raise ValueError(
            "member_ids must contain one unique non-empty string per member"
        )

    return {
        "energy": energy,
        "forces": forces,
        "stress": stress,
        "energy_reference": energy_reference,
        "forces_reference": forces_reference,
        "stress_reference": stress_reference,
        "n_atoms": n_atoms,
        "offsets": offsets,
        "member_ids": normalized_ids,
    }


def _pearson(left: torch.Tensor, right: torch.Tensor) -> float:
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = torch.linalg.vector_norm(left_centered) * torch.linalg.vector_norm(
        right_centered
    )
    return float(torch.dot(left_centered, right_centered).item() / denominator.item())


def _average_tied_ranks(values: torch.Tensor) -> torch.Tensor:
    flattened = values.detach().to(dtype=torch.float64).reshape(-1)
    order = sorted(
        range(flattened.numel()), key=lambda index: (float(flattened[index]), index)
    )
    ranks = torch.empty_like(flattened)
    start = 0
    while start < len(order):
        end = start + 1
        tied_value = float(flattened[order[start]])
        while end < len(order) and float(flattened[order[end]]) == tied_value:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average_rank
        start = end
    return ranks


def _correlation(
    uncertainty: torch.Tensor, error: torch.Tensor, constant_tolerance: float
) -> dict[str, object]:
    uncertainty_flat = uncertainty.detach().to(dtype=torch.float64).reshape(-1)
    error_flat = error.detach().to(dtype=torch.float64).reshape(-1)
    if uncertainty_flat.shape != error_flat.shape or uncertainty_flat.numel() == 0:
        raise ValueError("correlation inputs must have the same non-empty scalar shape")
    uncertainty_constant = bool(
        (uncertainty_flat.max() - uncertainty_flat.min()).item() <= constant_tolerance
    )
    error_constant = bool(
        (error_flat.max() - error_flat.min()).item() <= constant_tolerance
    )
    if uncertainty_constant or error_constant:
        return {
            "status": "undefined_constant_input",
            "pearson": None,
            "spearman": None,
            "uncertainty_constant": uncertainty_constant,
            "error_constant": error_constant,
            "n": uncertainty_flat.numel(),
        }
    return {
        "status": "ok",
        "pearson": _pearson(uncertainty_flat, error_flat),
        "spearman": _pearson(
            _average_tied_ranks(uncertainty_flat), _average_tied_ranks(error_flat)
        ),
        "uncertainty_constant": False,
        "error_constant": False,
        "n": uncertainty_flat.numel(),
    }


def _risk_coverage(
    uncertainty: torch.Tensor, error: torch.Tensor, coverages: tuple[float, ...]
) -> list[dict[str, float | int]]:
    uncertainty_flat = uncertainty.detach().reshape(-1)
    error_flat = error.detach().reshape(-1)
    if uncertainty_flat.shape != error_flat.shape or uncertainty_flat.numel() == 0:
        raise ValueError("risk inputs must have the same non-empty scalar shape")
    order = sorted(
        range(uncertainty_flat.numel()),
        key=lambda index: (float(uncertainty_flat[index]), index),
    )
    rows: list[dict[str, float | int]] = []
    for coverage in coverages:
        kept_count = max(1, math.ceil(len(order) * coverage))
        kept = torch.tensor(order[:kept_count], dtype=torch.int64)
        rows.append(
            {
                "coverage": coverage,
                "kept": kept_count,
                "risk": float(error_flat[kept].mean().item()),
            }
        )
    return rows


def evaluate_prediction(
    payload: Mapping[str, object],
    coverages: Sequence[float],
    constant_tolerance: float,
) -> EvaluationArtifacts:
    """Evaluate an in-memory canonical prediction without model, data, or I/O."""
    if isinstance(constant_tolerance, bool) or not isinstance(
        constant_tolerance, (float, int)
    ):
        raise TypeError("constant_tolerance must be a real number")
    tolerance = float(constant_tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("constant_tolerance must be finite and positive")
    coverage_values = _validate_coverages(coverages)
    data = _validate_payload(payload)

    energy = data["energy"]
    forces = data["forces"]
    stress = data["stress"]
    energy_reference = data["energy_reference"]
    forces_reference = data["forces_reference"]
    stress_reference = data["stress_reference"]
    n_atoms = data["n_atoms"]
    offsets = data["offsets"]
    n_atoms_float = n_atoms.to(dtype=energy.dtype)

    energy_mean = energy.to(dtype=torch.float64).mean(dim=0).to(dtype=energy.dtype)
    forces_mean = forces.to(dtype=torch.float64).mean(dim=0).to(dtype=forces.dtype)
    stress_mean = stress.to(dtype=torch.float64).mean(dim=0).to(dtype=stress.dtype)
    ensemble = {"energy": energy_mean, "forces": forces_mean, "stress": stress_mean}
    _assert_finite_derived(ensemble, "ensemble")

    energy_per_atom = energy / n_atoms_float.unsqueeze(0)
    force_deviation = forces - forces_mean.unsqueeze(0)
    force_vector_std = torch.sqrt(
        torch.mean(torch.sum(force_deviation**2, dim=-1), dim=0)
    )
    force_vector_gmd = vector_gmd(forces)
    force_structure_std = reduce_force_by_structure(force_vector_std, offsets)
    energy_total_uncertainty = {
        "std": population_std(energy),
        "gmd": scalar_gmd(energy),
    }
    energy_per_atom_uncertainty = {
        "std": population_std(energy_per_atom),
        "gmd": scalar_gmd(energy_per_atom),
    }
    force_component_uncertainty = {
        "std": population_std(forces),
        "gmd": scalar_gmd(forces),
    }
    uncertainty = {
        "formula_version": FORMULA_VERSION,
        "energy_total": energy_total_uncertainty,
        "energy_per_atom": energy_per_atom_uncertainty,
        "force_component": force_component_uncertainty,
        "force_atom_vector": {"std": force_vector_std, "gmd": force_vector_gmd},
        "force_structure": {"std": force_structure_std},
    }
    _assert_finite_derived(uncertainty, "uncertainty")

    energy_mean_float64 = energy_mean.to(dtype=torch.float64)
    forces_mean_float64 = forces_mean.to(dtype=torch.float64)
    stress_mean_float64 = stress_mean.to(dtype=torch.float64)
    energy_reference_float64 = energy_reference.to(dtype=torch.float64)
    forces_reference_float64 = forces_reference.to(dtype=torch.float64)
    stress_reference_float64 = stress_reference.to(dtype=torch.float64)
    n_atoms_float64 = n_atoms.to(dtype=torch.float64)
    energy_total_error = torch.abs(energy_mean_float64 - energy_reference_float64)
    energy_per_atom_error = torch.abs(
        energy_mean_float64 / n_atoms_float64
        - energy_reference_float64 / n_atoms_float64
    )
    force_component_error = torch.abs(forces_mean_float64 - forces_reference_float64)
    force_vector_error = torch.linalg.vector_norm(
        forces_mean_float64 - forces_reference_float64, dim=-1
    )
    force_structure_error = reduce_force_by_structure(force_vector_error, offsets)
    stress_component_error = torch.abs(stress_mean_float64 - stress_reference_float64)
    _assert_finite_derived(
        {
            "energy_total": energy_total_error,
            "energy_per_atom": energy_per_atom_error,
            "force_component": force_component_error,
            "force_vector": force_vector_error,
            "force_structure": force_structure_error,
            "stress_component": stress_component_error,
        },
        "errors",
    )

    correlation_inputs = {
        "energy_total_std": (energy_total_uncertainty["std"], energy_total_error),
        "energy_total_gmd": (energy_total_uncertainty["gmd"], energy_total_error),
        "energy_per_atom_std": (
            energy_per_atom_uncertainty["std"],
            energy_per_atom_error,
        ),
        "energy_per_atom_gmd": (
            energy_per_atom_uncertainty["gmd"],
            energy_per_atom_error,
        ),
        "force_component_std": (
            force_component_uncertainty["std"],
            force_component_error,
        ),
        "force_component_gmd": (
            force_component_uncertainty["gmd"],
            force_component_error,
        ),
        "force_atom_vector_std": (force_vector_std, force_vector_error),
        "force_atom_vector_gmd": (force_vector_gmd, force_vector_error),
    }
    for reduction in ("mean", "max", "q95"):
        correlation_inputs[f"force_structure_{reduction}_std"] = (
            force_structure_std[reduction],
            force_structure_error[reduction],
        )
    correlations = {
        name: _correlation(disagreement, error, tolerance)
        for name, (disagreement, error) in correlation_inputs.items()
    }
    risks = {
        "energy_total_std": _risk_coverage(
            energy_total_uncertainty["std"], energy_total_error, coverage_values
        ),
        "energy_per_atom_std": _risk_coverage(
            energy_per_atom_uncertainty["std"],
            energy_per_atom_error,
            coverage_values,
        ),
        "force_atom_vector_std": _risk_coverage(
            force_vector_std, force_vector_error, coverage_values
        ),
        "force_structure_q95_std": _risk_coverage(
            force_structure_std["q95"],
            force_structure_error["q95"],
            coverage_values,
        ),
    }
    metrics = {
        "schema_version": 4,
        "mae": {
            "energy_total": global_mae(energy_mean, energy_reference),
            "energy_per_atom": global_mae(
                energy_mean_float64 / n_atoms_float64,
                energy_reference_float64 / n_atoms_float64,
            ),
            "force_component": global_mae(forces_mean, forces_reference),
            "stress_component": global_mae(stress_mean, stress_reference),
        },
        "counts": {
            "structures": energy_mean.numel(),
            "atoms": forces_mean.shape[0],
            "force_components": force_component_error.numel(),
            "stress_components": stress_component_error.numel(),
        },
        "correlations": correlations,
        "risk_coverage": risks,
    }
    diagnostics = list(correlations.values())
    report_inputs = {
        "formula_version": FORMULA_VERSION,
        "metric_schema_version": 4,
        "member_count": len(data["member_ids"]),
        "structure_count": energy_mean.numel(),
        "atom_count": forces_mean.shape[0],
        "coverages": list(coverage_values),
        "undefined_correlation_count": sum(
            diagnostic["status"] == "undefined_constant_input"
            for diagnostic in diagnostics
        ),
    }
    _assert_finite_derived(metrics, "metrics")
    _assert_finite_derived(report_inputs, "report_inputs")
    return EvaluationArtifacts(
        ensemble=ensemble,
        uncertainty=uncertainty,
        metrics=metrics,
        report_inputs=report_inputs,
    )


def evaluate_fge(config: object) -> Path:
    """Evaluate only the stored canonical prediction and publish formal artifacts."""
    try:
        layout = ExperimentLayout(Path(config.paths.output_root) / config.project.name)
        coverages = config.evaluation.risk_coverages
        tolerance = config.evaluation.constant_tolerance
    except AttributeError as exc:
        raise HardFailure("evaluation requires an FGE configuration") from exc
    try:
        payload = torch.load(
            layout.prediction_tensor, weights_only=True, map_location="cpu"
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise HardFailure("canonical prediction cannot be loaded") from exc
    if not isinstance(payload, Mapping):
        raise HardFailure("canonical prediction is not a mapping")
    try:
        result = evaluate_prediction(payload, coverages, tolerance)
    except (TypeError, ValueError) as exc:
        raise HardFailure("canonical prediction cannot be evaluated") from exc
    directory = layout.evaluation_dir / "legacy_equal_weight"
    if directory.exists():
        raise HardFailure("formal evaluation output is immutable")
    with sibling_staging(directory) as staging:
        atomic_torch_save(staging / "ensemble.pt", dict(result.ensemble))
        atomic_torch_save(staging / "uncertainty.pt", dict(result.uncertainty))
        atomic_write_json(staging / "metrics.json", dict(result.metrics))
        (staging / "report.md").write_text(
            "# Canonical FGE report\n"
            + json.dumps(dict(result.report_inputs), sort_keys=True)
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
    return directory
