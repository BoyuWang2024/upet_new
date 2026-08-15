"""Validation-only Alpha calibration and independent eta selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .artifacts import (
    RunPaths,
    atomic_json_dump,
    load_verified_manifest,
    sha256_file,
    stage_identity,
)
from .calibration_progress import (
    load_calibration_progress,
    save_calibration_progress,
)
from .checkpoint import load_checkpoint
from .config import LLPRConfig
from .curvature import resolve_curvature_stage
from .data import build_system, dataset_identity, iter_samples
from .observables import compute_structure_jacobians
from .readout import discover_readout_layout
from .reuse import materialize_reused_stage, validate_curvature_layout
from .ridge import (
    RidgeCandidate,
    fit_candidates,
    fixed_candidate,
    prepare_spectrum,
    quadratic_forms,
)


@dataclass(frozen=True)
class CalibrationRecord:
    target: str
    eta: float
    alpha: float
    alpha_sq: float
    gaussian_nll: float
    condition_number: float
    condition_warning: bool
    count: int
    coverage_1sigma: float
    coverage_2sigma: float
    coverage_3sigma: float


def moment_alpha(residuals: torch.Tensor, q: torch.Tensor) -> float:
    residuals = residuals.detach().reshape(-1).to(torch.float64)
    q = q.detach().reshape(-1).to(torch.float64)
    if residuals.shape != q.shape or residuals.numel() == 0:
        raise ValueError("residuals and q must have the same non-empty shape")
    if not bool(torch.isfinite(residuals).all()):
        raise ValueError("residuals must be finite")
    if not bool(torch.isfinite(q).all()) or bool((q <= 0).any()):
        raise ValueError("q must be finite and strictly positive")
    alpha_sq = torch.mean(residuals.square() / q)
    if not bool(torch.isfinite(alpha_sq)) or float(alpha_sq) <= 0:
        raise ValueError("Alpha squared must be finite and strictly positive")
    return float(torch.sqrt(alpha_sq))


def gaussian_nll(residuals: torch.Tensor, variance: torch.Tensor) -> float:
    residuals = residuals.detach().reshape(-1).to(torch.float64)
    variance = variance.detach().reshape(-1).to(torch.float64)
    if residuals.shape != variance.shape or residuals.numel() == 0:
        raise ValueError("residuals and variance must have matching non-empty shapes")
    if not bool(torch.isfinite(variance).all()) or bool((variance <= 0).any()):
        raise ValueError("variance must be finite and strictly positive")
    values = 0.5 * (torch.log(2 * torch.pi * variance) + residuals.square() / variance)
    return float(torch.mean(values))


def select_candidate(
    records: list[CalibrationRecord] | tuple[CalibrationRecord, ...],
) -> CalibrationRecord:
    if not records:
        raise ValueError("calibration records must not be empty")
    targets = {record.target for record in records}
    if len(targets) != 1:
        raise ValueError("eta selection must contain exactly one target")
    return min(records, key=lambda record: (record.gaussian_nll, record.eta))


def _record(
    candidate: RidgeCandidate,
    residuals: torch.Tensor,
    q: torch.Tensor,
) -> CalibrationRecord:
    alpha = moment_alpha(residuals, q)
    variance = alpha**2 * q
    std = torch.sqrt(variance)
    standardized = residuals.abs() / std
    return CalibrationRecord(
        target=candidate.target,
        eta=candidate.eta,
        alpha=alpha,
        alpha_sq=alpha**2,
        gaussian_nll=gaussian_nll(residuals, variance),
        condition_number=candidate.condition_number,
        condition_warning=candidate.condition_warning,
        count=residuals.numel(),
        coverage_1sigma=float(torch.mean((standardized <= 1).to(torch.float64))),
        coverage_2sigma=float(torch.mean((standardized <= 2).to(torch.float64))),
        coverage_3sigma=float(torch.mean((standardized <= 3).to(torch.float64))),
    )


def _candidates(
    config: LLPRConfig,
    energy_matrix: torch.Tensor,
    force_matrix: torch.Tensor,
) -> dict[str, tuple[RidgeCandidate, ...]]:
    spectra = {
        "energy": prepare_spectrum(energy_matrix),
        "force": prepare_spectrum(force_matrix),
    }
    ridge = config.calibration.ridge
    if ridge.mode == "fixed":
        assert ridge.eta is not None
        return {
            "energy": (
                fixed_candidate(
                    "energy",
                    ridge.eta.energy,
                    spectra["energy"],
                    ridge.max_condition_number,
                ),
            ),
            "force": (
                fixed_candidate(
                    "force",
                    ridge.eta.force,
                    spectra["force"],
                    ridge.max_condition_number,
                ),
            ),
        }
    assert ridge.fit is not None
    return {
        target: fit_candidates(
            target,
            spectra[target],
            ridge.fit.candidate_multipliers,
            ridge.max_condition_number,
        )
        for target in ("energy", "force")
    }


def resolve_calibration_stage(config: LLPRConfig) -> Path:
    """Return a verified calibration stage, computing it only when needed."""
    curvature_dir = resolve_curvature_stage(config)
    curvature_manifest = load_verified_manifest(
        curvature_dir / "manifest.json", verify_npz=True
    )
    reuse = config.reuse.calibration if config.reuse is not None else None
    if reuse is None:
        return run_calibrate(config)
    root = config.output.root / config.output.experiment
    return materialize_reused_stage(
        reuse.path,
        RunPaths(root).calibration,
        stage="calibration",
        identity=reuse.identity,
        expected_payload={
            "curvature_identity": curvature_manifest["identity"],
            "ridge": config.calibration.ridge.model_dump(mode="json"),
        },
        expected_curvature_identity=str(curvature_manifest["identity"]),
    )


def run_calibrate(config: LLPRConfig) -> Path:
    """Calibrate Alpha and optionally eta using validation data only."""
    curvature_dir = resolve_curvature_stage(config)
    curvature_manifest = load_verified_manifest(
        curvature_dir / "manifest.json", verify_npz=True
    )
    calibration_path = config.data.calibration
    if calibration_path is None:
        raise ValueError("numerical calibration requires calibration data")
    validation = dataset_identity(calibration_path)
    if (
        config.data.calibration_expected_sha256 is not None
        and validation.sha256 != config.data.calibration_expected_sha256
    ):
        raise ValueError("calibration dataset SHA mismatch")
    identity = stage_identity(
        "calibration",
        {
            "curvature_identity": curvature_manifest["identity"],
            "validation_sha256": validation.sha256,
            "ridge": config.calibration.ridge.model_dump(mode="json"),
        },
    )
    identity_value = str(identity["identity"])
    root = config.output.root / config.output.experiment
    stage_dir = RunPaths(root).calibration / identity_value
    manifest_path = stage_dir / "manifest.json"
    if manifest_path.exists():
        load_verified_manifest(manifest_path, {"identity": identity_value})
        return stage_dir
    stage_dir.mkdir(parents=True, exist_ok=True)

    with np.load(curvature_dir / "curvature.npz", allow_pickle=False) as archive:
        energy_matrix = torch.from_numpy(archive["energy"].copy()).to(torch.float64)
        force_matrix = torch.from_numpy(archive["force"].copy()).to(torch.float64)
    candidates = _candidates(config, energy_matrix, force_matrix)
    device = torch.device(config.runtime.device)
    loaded = load_checkpoint(config.checkpoint, device=device, dtype=torch.float64)
    layout = discover_readout_layout(loaded.model)
    validate_curvature_layout(curvature_dir, curvature_manifest, layout)
    progress_path = stage_dir / "progress.npz"
    candidate_counts = {
        target: len(target_candidates)
        for target, target_candidates in candidates.items()
    }
    if progress_path.exists():
        (
            energy_residuals,
            force_residuals,
            q_values,
            next_index,
        ) = load_calibration_progress(
            progress_path,
            expected_identity=identity_value,
            candidate_counts=candidate_counts,
            expected_structure_count=validation.structure_count,
        )
    else:
        energy_residuals = []
        force_residuals = []
        q_values = {
            target: [[] for _ in target_candidates]
            for target, target_candidates in candidates.items()
        }
        next_index = 0
    for sample in iter_samples(calibration_path):
        if sample.index < next_index:
            continue
        system = build_system(sample, loaded.model, device=device, dtype=torch.float64)
        jacobians = compute_structure_jacobians(
            loaded.model,
            system,
            layout,
            config.runtime.jacobian_backend,
            config.runtime.force_component_chunk_size,
        )
        energy_residual = (
            jacobians.energy_pred_per_atom
            - sample.energy_reference_total / len(sample.atoms)
        ).reshape(1)
        force_residual = jacobians.force_pred - torch.from_numpy(
            sample.force_reference.reshape(-1)
        )
        energy_residuals.append(energy_residual)
        force_residuals.append(force_residual)
        for target, gradients in (
            ("energy", jacobians.energy_jacobian.reshape(1, -1)),
            ("force", jacobians.force_jacobian),
        ):
            for index, candidate in enumerate(candidates[target]):
                q_values[target][index].append(
                    quadratic_forms(candidate.cholesky, gradients)
                )

        next_index = sample.index + 1
        if next_index % config.curvature.checkpoint_interval == 0:
            save_calibration_progress(
                progress_path,
                identity=identity_value,
                next_structure_index=next_index,
                energy_residuals=energy_residuals,
                force_residuals=force_residuals,
                q_values=q_values,
            )

    save_calibration_progress(
        progress_path,
        identity=identity_value,
        next_structure_index=next_index,
        energy_residuals=energy_residuals,
        force_residuals=force_residuals,
        q_values=q_values,
    )

    residuals = {
        "energy": torch.cat(energy_residuals),
        "force": torch.cat(force_residuals),
    }
    records: dict[str, list[CalibrationRecord]] = {}
    for target in ("energy", "force"):
        records[target] = [
            _record(
                candidate,
                residuals[target],
                torch.cat(q_values[target][index]),
            )
            for index, candidate in enumerate(candidates[target])
        ]
    selected = {
        target: select_candidate(records[target]) for target in ("energy", "force")
    }
    candidates_path = stage_dir / "candidates.json"
    summary_path = stage_dir / "summary.json"
    atomic_json_dump(
        candidates_path,
        {
            target: [asdict(record) for record in records[target]]
            for target in ("energy", "force")
        },
    )
    atomic_json_dump(
        summary_path,
        {
            "mode": config.calibration.ridge.mode,
            "selected": {
                target: asdict(selected[target]) for target in ("energy", "force")
            },
        },
    )
    atomic_json_dump(
        manifest_path,
        {
            **identity,
            "status": "complete",
            "curvature_identity": curvature_manifest["identity"],
            "files": {
                candidates_path.name: sha256_file(candidates_path),
                summary_path.name: sha256_file(summary_path),
            },
        },
    )
    progress_path.unlink(missing_ok=True)
    return stage_dir
