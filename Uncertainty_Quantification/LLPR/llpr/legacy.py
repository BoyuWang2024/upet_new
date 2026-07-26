"""Strict, model-free migration of the audited legacy LLPR result tree."""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field

from .artifacts import (
    FORMULA_VERSION,
    SCHEMA_VERSION,
    atomic_json_dump,
    atomic_npz_save,
    sha256_file,
    stable_id,
    stage_identity,
    verify_run,
)
from .calibration import gaussian_nll
from .inference import summarize_evaluation


Classification = Literal["authoritative", "legacy_smoke", "incomplete", "orphan"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LegacyTargets(StrictModel):
    energy: float = Field(gt=0)
    force: float = Field(gt=0)


class LegacyDimensions(StrictModel):
    energy: int = Field(gt=0)
    force: int = Field(gt=0)
    total: int = Field(gt=0)


class LegacyCounts(StrictModel):
    structures: int = Field(gt=0)
    atoms: int = Field(gt=0)
    force_components: int = Field(gt=0)


class LegacyImportConfig(StrictModel):
    source_root: Path
    destination_root: Path
    experiment: str = Field(min_length=1)
    checkpoint_path: Path
    build_path: Path
    validation_path: Path
    test_path: Path
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    build_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    test_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_eta: LegacyTargets
    expected_alpha: LegacyTargets
    expected_dimensions: LegacyDimensions
    expected_counts: LegacyCounts


@dataclass(frozen=True)
class InventoryEntry:
    source_absolute: str
    destination_relative: str
    bytes: int
    sha256: str
    classification: Classification
    formal_source: bool


def _classification(relative: str) -> Classification:
    if relative.endswith("results/llpr_test_details.npz") or "dry_run" in relative:
        return "legacy_smoke"
    if relative.endswith("reliability_matpes_linear_fit_summary.json"):
        return "incomplete"
    if relative.endswith("reliability_matpes_llpr.png"):
        return "orphan"
    return "authoritative"


def _is_formal_source(relative: str) -> bool:
    names = {
        "results/H_E_full_run.npz",
        "results/H_F_full_run.npz",
        "results/H_EF_full_run.npz",
        "results/LLPR/llpr_test_full_gpu_details.npz",
        "results/LLPR/llpr_test_full_gpu_summary.json",
    }
    return relative in names


def build_source_inventory(source: Path) -> tuple[InventoryEntry, ...]:
    source = Path(source).resolve()
    if not source.is_dir():
        raise ValueError(f"legacy source root does not exist: {source}")
    entries: list[InventoryEntry] = []
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        relative = path.relative_to(source).as_posix()
        entries.append(
            InventoryEntry(
                source_absolute=str(path),
                destination_relative=relative,
                bytes=path.stat().st_size,
                sha256=sha256_file(path),
                classification=_classification(relative),
                formal_source=_is_formal_source(relative),
            )
        )
    if not entries:
        raise ValueError("legacy source inventory is empty")
    return tuple(entries)


def _load_matrix(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if "H" not in archive:
            raise ValueError(f"legacy matrix {path} has no H array")
        matrix = np.asarray(archive["H"], dtype=np.float64)
    return matrix


def validate_legacy_matrices(
    paths: dict[str, Path],
    *,
    energy_dim: int,
    force_dim: int,
) -> dict[str, np.ndarray]:
    total = energy_dim + force_dim
    matrices = {name: _load_matrix(path) for name, path in paths.items()}
    for name, matrix in matrices.items():
        if matrix.shape != (total, total):
            raise ValueError(f"{name} shape {matrix.shape} != {(total, total)}")
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"{name} contains non-finite values")
        if not np.allclose(matrix, matrix.T, rtol=0, atol=1.0e-12):
            raise ValueError(f"{name} is not symmetric")
    h_e = matrices["H_E"]
    h_f = matrices["H_F"]
    h_ef = matrices["H_EF"]
    if not np.allclose(h_e[energy_dim:, :], 0, rtol=0, atol=1.0e-12):
        raise ValueError("H_E has values outside the energy block")
    if not np.allclose(h_e[:, energy_dim:], 0, rtol=0, atol=1.0e-12):
        raise ValueError("H_E has values outside the energy block")
    if not np.allclose(h_f[:energy_dim, :], 0, rtol=0, atol=1.0e-12):
        raise ValueError("H_F has values outside the force block")
    if not np.allclose(h_f[:, :energy_dim], 0, rtol=0, atol=1.0e-12):
        raise ValueError("H_F has values outside the force block")
    if not np.allclose(h_ef, h_e + h_f, rtol=1.0e-12, atol=1.0e-14):
        raise ValueError("H_EF does not equal H_E + H_F")
    return {
        "energy": h_e[:energy_dim, :energy_dim].copy(),
        "force": h_f[energy_dim:, energy_dim:].copy(),
    }


def _assert_close(
    actual: np.ndarray,
    expected: np.ndarray,
    name: str,
) -> None:
    if not np.allclose(actual, expected, rtol=1.0e-12, atol=1.0e-14):
        difference = float(np.max(np.abs(actual - expected)))
        raise ValueError(f"legacy {name} invariant failed; max diff={difference}")


def validate_legacy_calibration(
    summary: dict[str, Any],
    config: LegacyImportConfig,
) -> None:
    checks = {
        "eta energy": (
            float(summary["damping_eta"]),
            config.expected_eta.energy,
        ),
        "eta force": (
            float(summary["damping_eta"]),
            config.expected_eta.force,
        ),
        "Alpha energy": (
            float(summary["alpha_energy"]),
            config.expected_alpha.energy,
        ),
        "Alpha force": (
            float(summary["alpha_force"]),
            config.expected_alpha.force,
        ),
    }
    mismatched = {
        name: values for name, values in checks.items() if values[0] != values[1]
    }
    if mismatched:
        raise ValueError(f"legacy Alpha/eta mismatch: {mismatched}")
    dimensions = config.expected_dimensions
    if (
        int(summary["dim_theta_E"]) != dimensions.energy
        or int(summary["dim_theta_F"]) != dimensions.force
        or int(summary["dim_theta_total"]) != dimensions.total
        or dimensions.energy + dimensions.force != dimensions.total
    ):
        raise ValueError("legacy readout dimensions mismatch")


def validate_legacy_evaluation(
    details: dict[str, np.ndarray],
    summary: dict[str, Any],
    config: LegacyImportConfig,
) -> None:
    counts = config.expected_counts
    structures = len(details["structure_index"])
    atoms = int(np.sum(details["num_atoms"]))
    components = len(details["force_residual"])
    if (structures, atoms, components) != (
        counts.structures,
        counts.atoms,
        counts.force_components,
    ):
        raise ValueError(
            f"legacy evaluation counts mismatch: {(structures, atoms, components)}"
        )
    if int(summary["num_structures_processed"]) != structures:
        raise ValueError("legacy summary structure count mismatch")
    if int(summary["num_structures_skipped"]) != 0:
        raise ValueError("legacy formal evaluation contains skipped structures")
    if int(summary["total_force_components"]) != components:
        raise ValueError("legacy summary force-component count mismatch")
    offsets = details["force_offsets"]
    if (
        offsets.shape != (structures + 1,)
        or int(offsets[0]) != 0
        or int(offsets[-1]) != components
        or np.any(np.diff(offsets) != 3 * details["num_atoms"])
    ):
        raise ValueError("legacy force offsets are inconsistent")
    _assert_close(
        details["energy_residual"],
        details["energy_pred_per_atom"] - details["energy_true_per_atom"],
        "energy residual",
    )
    _assert_close(
        details["force_residual"],
        details["force_pred"] - details["force_true"],
        "force residual",
    )
    _assert_close(
        details["energy_calibrated_var"],
        config.expected_alpha.energy**2 * details["energy_raw_var"],
        "energy calibrated variance",
    )
    _assert_close(
        details["force_calibrated_var_component"],
        config.expected_alpha.force**2 * details["force_raw_var_component"],
        "force calibrated variance",
    )
    _assert_close(
        details["energy_calibrated_std"],
        np.sqrt(details["energy_calibrated_var"]),
        "energy calibrated std",
    )
    _assert_close(
        details["force_calibrated_std_component"],
        np.sqrt(details["force_calibrated_var_component"]),
        "force calibrated std",
    )
    _assert_close(
        details["energy_inverse_variance"],
        1.0 / details["energy_calibrated_var"],
        "energy inverse variance",
    )
    _assert_close(
        details["force_inverse_variance_component"],
        1.0 / details["force_calibrated_var_component"],
        "force inverse variance",
    )


def _load_config(path: Path) -> LegacyImportConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("legacy import config must be a mapping")
    for name in (
        "source_root",
        "destination_root",
        "checkpoint_path",
        "build_path",
        "validation_path",
        "test_path",
    ):
        raw[name] = Path(raw[name]).expanduser().resolve()
    return LegacyImportConfig.model_validate(raw)


def _load_details(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def _coverage(residual: np.ndarray, variance: np.ndarray, level: int) -> float:
    return float(np.mean(np.abs(residual) <= level * np.sqrt(variance)))


def _calibration_record(
    target: str,
    eta: float,
    alpha: float,
    residual: np.ndarray,
    raw_variance: np.ndarray,
    condition_number: float,
) -> dict[str, object]:
    variance = alpha**2 * raw_variance
    return {
        "target": target,
        "eta": eta,
        "alpha": alpha,
        "alpha_sq": alpha**2,
        "gaussian_nll": gaussian_nll(
            torch_from_numpy(residual), torch_from_numpy(variance)
        ),
        "condition_number": condition_number,
        "condition_warning": condition_number > 1.0e10,
        "count": len(residual),
        "coverage_1sigma": _coverage(residual, variance, 1),
        "coverage_2sigma": _coverage(residual, variance, 2),
        "coverage_3sigma": _coverage(residual, variance, 3),
    }


def torch_from_numpy(value: np.ndarray):
    import torch

    return torch.from_numpy(np.asarray(value, dtype=np.float64))


def _condition_number(matrix: np.ndarray, eta: float) -> float:
    eigenvalues = np.linalg.eigvalsh((matrix + matrix.T) / 2)
    return float((float(eigenvalues[-1]) + eta) / (float(eigenvalues[0]) + eta))


def _copy_inventory(
    source: Path,
    destination: Path,
    inventory: tuple[InventoryEntry, ...],
) -> None:
    for entry in inventory:
        target = destination / entry.destination_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / entry.destination_relative, target)
        if sha256_file(target) != entry.sha256:
            raise ValueError(f"legacy raw copy hash mismatch: {target}")


def _validate_input_identities(config: LegacyImportConfig) -> None:
    inputs = {
        "checkpoint": (config.checkpoint_path, config.checkpoint_sha256),
        "build": (config.build_path, config.build_sha256),
        "validation": (config.validation_path, config.validation_sha256),
        "test": (config.test_path, config.test_sha256),
    }
    for name, (path, expected) in inputs.items():
        if not path.is_file():
            raise ValueError(f"{name} identity input does not exist: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"{name} SHA mismatch: configured {expected}, actual {actual}"
            )


def import_legacy(config_path: Path) -> Path:
    """Audit, convert, verify, and atomically publish legacy artifacts."""
    config = _load_config(config_path)
    _validate_input_identities(config)
    inventory = build_source_inventory(config.source_root)
    inventory_payload = [asdict(entry) for entry in inventory]
    inventory_hash = stable_id(inventory_payload, length=64)
    root_identity = stage_identity(
        "legacy_import",
        {
            "inventory_sha256": inventory_hash,
            "checkpoint_sha256": config.checkpoint_sha256,
            "build_sha256": config.build_sha256,
            "validation_sha256": config.validation_sha256,
            "test_sha256": config.test_sha256,
            "eta": config.expected_eta.model_dump(),
            "alpha": config.expected_alpha.model_dump(),
            "dimensions": config.expected_dimensions.model_dump(),
            "counts": config.expected_counts.model_dump(),
        },
    )
    destination = config.destination_root / config.experiment
    if destination.exists():
        manifest_path = destination / "manifest.json"
        if manifest_path.is_file():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("identity") == root_identity["identity"]:
                verify_run(destination, level="full")
                return destination
        raise ValueError(f"legacy import identity collision at {destination}")

    staging = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)
    try:
        raw_destination = staging / "legacy_raw"
        _copy_inventory(config.source_root, raw_destination, inventory)
        inventory_path = staging / "inventory.json"
        atomic_json_dump(
            inventory_path,
            {
                "inventory_sha256": inventory_hash,
                "files": inventory_payload,
            },
        )

        results = config.source_root / "results"
        matrices = validate_legacy_matrices(
            {
                "H_E": results / "H_E_full_run.npz",
                "H_F": results / "H_F_full_run.npz",
                "H_EF": results / "H_EF_full_run.npz",
            },
            energy_dim=config.expected_dimensions.energy,
            force_dim=config.expected_dimensions.force,
        )
        legacy_summary_path = results / "LLPR/llpr_test_full_gpu_summary.json"
        legacy_details_path = results / "LLPR/llpr_test_full_gpu_details.npz"
        legacy_summary = json.loads(legacy_summary_path.read_text(encoding="utf-8"))
        details = _load_details(legacy_details_path)
        validate_legacy_calibration(legacy_summary, config)
        validate_legacy_evaluation(details, legacy_summary, config)

        curvature_identity = stage_identity(
            "curvature",
            {
                "origin": "legacy_import",
                "root_identity": root_identity["identity"],
                "checkpoint_sha256": config.checkpoint_sha256,
                "build_sha256": config.build_sha256,
            },
        )
        curvature_dir = staging / "curvature" / str(curvature_identity["identity"])
        curvature_path = curvature_dir / "curvature.npz"
        diagnostics_path = curvature_dir / "diagnostics.json"
        atomic_npz_save(curvature_path, matrices)
        condition_energy = _condition_number(
            matrices["energy"], config.expected_eta.energy
        )
        condition_force = _condition_number(
            matrices["force"], config.expected_eta.force
        )
        atomic_json_dump(
            diagnostics_path,
            {
                "energy_dimension": config.expected_dimensions.energy,
                "force_dimension": config.expected_dimensions.force,
                "total_dimension": config.expected_dimensions.total,
                "structure_count": config.expected_counts.structures,
                "atom_count": config.expected_counts.atoms,
                "force_component_count": config.expected_counts.force_components,
                "condition_number_energy_h_plus_eta_i": condition_energy,
                "condition_number_force_h_plus_eta_i": condition_force,
                "legacy_eta_squared_diagnostic": "raw provenance only",
            },
        )
        atomic_json_dump(
            curvature_dir / "manifest.json",
            {
                **curvature_identity,
                "status": "complete",
                "origin": "legacy_import",
                "legacy_fixed_ridge": True,
                "files": {
                    curvature_path.name: sha256_file(curvature_path),
                    diagnostics_path.name: sha256_file(diagnostics_path),
                },
            },
        )

        calibration_identity = stage_identity(
            "calibration",
            {
                "origin": "legacy_import",
                "curvature_identity": curvature_identity["identity"],
                "validation_sha256": config.validation_sha256,
                "mode": "fixed",
                "eta": config.expected_eta.model_dump(),
            },
        )
        calibration_dir = (
            staging / "calibration" / str(calibration_identity["identity"])
        )
        energy_record = _calibration_record(
            "energy",
            config.expected_eta.energy,
            config.expected_alpha.energy,
            details["energy_residual"],
            details["energy_raw_var"],
            condition_energy,
        )
        force_record = _calibration_record(
            "force",
            config.expected_eta.force,
            config.expected_alpha.force,
            details["force_residual"],
            details["force_raw_var_component"],
            condition_force,
        )
        candidates_path = calibration_dir / "candidates.json"
        calibration_summary_path = calibration_dir / "summary.json"
        atomic_json_dump(
            candidates_path,
            {"energy": [energy_record], "force": [force_record]},
        )
        atomic_json_dump(
            calibration_summary_path,
            {
                "mode": "fixed",
                "selected": {
                    "energy": energy_record,
                    "force": force_record,
                },
            },
        )
        atomic_json_dump(
            calibration_dir / "manifest.json",
            {
                **calibration_identity,
                "status": "complete",
                "origin": "legacy_import",
                "legacy_fixed_ridge": True,
                "files": {
                    candidates_path.name: sha256_file(candidates_path),
                    calibration_summary_path.name: sha256_file(
                        calibration_summary_path
                    ),
                },
            },
        )

        evaluation_identity = stage_identity(
            "evaluation",
            {
                "origin": "legacy_import",
                "curvature_identity": curvature_identity["identity"],
                "calibration_identity": calibration_identity["identity"],
                "test_sha256": config.test_sha256,
            },
        )
        evaluation_dir = (
            staging
            / "evaluation"
            / str(calibration_identity["identity"])
            / str(evaluation_identity["identity"])
        )
        details_path = evaluation_dir / "details.npz"
        summary_path = evaluation_dir / "summary.json"
        preview_path = evaluation_dir / "preview.json"
        atomic_npz_save(details_path, details)
        canonical_summary = summarize_evaluation(details)
        canonical_summary.update(
            {
                "alpha_energy": config.expected_alpha.energy,
                "alpha_force": config.expected_alpha.force,
                "eta_energy": config.expected_eta.energy,
                "eta_force": config.expected_eta.force,
            }
        )
        atomic_json_dump(summary_path, canonical_summary)
        atomic_json_dump(
            preview_path,
            {
                "summary": canonical_summary,
                "first_structure_indices": details["structure_index"][:10].tolist(),
            },
        )
        atomic_json_dump(
            evaluation_dir / "manifest.json",
            {
                **evaluation_identity,
                "status": "complete",
                "origin": "legacy_import",
                "legacy_fixed_ridge": True,
                "files": {
                    details_path.name: sha256_file(details_path),
                    summary_path.name: sha256_file(summary_path),
                    preview_path.name: sha256_file(preview_path),
                },
            },
        )

        atomic_json_dump(
            staging / "manifest.json",
            {
                **root_identity,
                "status": "complete",
                "origin": "legacy_import",
                "legacy_fixed_ridge": True,
                "schema_version": SCHEMA_VERSION,
                "formula_version": FORMULA_VERSION,
                "curvature_identity": curvature_identity["identity"],
                "calibration_identity": calibration_identity["identity"],
                "evaluation_identity": evaluation_identity["identity"],
                "files": {
                    inventory_path.name: sha256_file(inventory_path),
                },
            },
        )
        verify_run(staging, level="full")
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(destination)
        return destination
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
