"""Test evaluation, transactional shards, and canonical aggregation."""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from dataclasses import fields
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from .artifacts import (
    RunPaths,
    atomic_json_dump,
    atomic_npz_save,
    load_complete_manifest,
    sha256_file,
    stage_identity,
)
from .calibration import (
    CalibrationRecord,
    run_calibrate,
)
from .checkpoint import load_checkpoint
from .config import LLPRConfig
from .curvature import run_build
from .data import LLPRSample, build_system, dataset_identity, iter_samples
from .observables import StructureJacobians, compute_structure_jacobians
from .readout import discover_readout_layout
from .ridge import quadratic_forms


STRUCTURE_FIELDS = {
    "structure_index",
    "num_atoms",
    "energy_pred_total",
    "energy_true_total",
    "energy_pred_per_atom",
    "energy_true_per_atom",
    "energy_residual",
    "energy_raw_var",
    "energy_calibrated_var",
    "energy_calibrated_std",
    "energy_inverse_variance",
    "energy_raw_var_total_derived",
    "energy_calibrated_var_total_derived",
    "force_raw_var_component_mean_structure",
    "force_calibrated_var_component_mean_structure",
    "force_calibrated_std_component_rms_structure",
}
ATOM_FIELDS = {
    "force_raw_var_atom_mean",
    "force_calibrated_var_atom_mean",
    "force_calibrated_std_atom_rms",
}
FORCE_FIELDS = {
    "force_structure_index",
    "force_component_index_within_structure",
    "force_atom_index",
    "force_cartesian_index",
    "force_pred",
    "force_true",
    "force_residual",
    "force_raw_var_component",
    "force_calibrated_var_component",
    "force_calibrated_std_component",
    "force_inverse_variance_component",
}


def _positive_q(
    cholesky: torch.Tensor,
    gradients: torch.Tensor,
    *,
    structure_index: int,
    target: str,
) -> torch.Tensor:
    try:
        return quadratic_forms(cholesky, gradients)
    except ValueError as error:
        raise ValueError(
            f"structure {structure_index} {target}: invalid quadratic form"
        ) from error


def evaluate_structure(
    sample: LLPRSample,
    jacobians: StructureJacobians,
    calibration: Mapping[str, CalibrationRecord],
    solvers: Mapping[str, torch.Tensor],
) -> dict[str, np.ndarray]:
    """Evaluate one complete structure without recalibrating Alpha."""
    atom_count = len(sample.atoms)
    energy_q = _positive_q(
        solvers["energy"],
        jacobians.energy_jacobian.reshape(1, -1),
        structure_index=sample.index,
        target="energy",
    )
    force_q = _positive_q(
        solvers["force"],
        jacobians.force_jacobian,
        structure_index=sample.index,
        target="force",
    )
    energy_raw = energy_q.numpy()
    force_raw = force_q.numpy()
    energy_alpha_sq = calibration["energy"].alpha_sq
    force_alpha_sq = calibration["force"].alpha_sq
    energy_var = energy_alpha_sq * energy_raw
    force_var = force_alpha_sq * force_raw
    if (
        not np.all(np.isfinite(energy_var))
        or not np.all(energy_var > 0)
        or not np.all(np.isfinite(force_var))
        or not np.all(force_var > 0)
    ):
        raise ValueError(f"structure {sample.index}: invalid calibrated variance")

    energy_pred_total = float(jacobians.energy_pred_total)
    energy_pred_per_atom = float(jacobians.energy_pred_per_atom)
    energy_true_total = float(sample.energy_reference_total)
    energy_true_per_atom = energy_true_total / atom_count
    force_pred = jacobians.force_pred.numpy()
    force_true = sample.force_reference.reshape(-1)
    component_count = 3 * atom_count
    force_atom_raw = force_raw.reshape(atom_count, 3).mean(axis=1)
    force_atom_var = force_var.reshape(atom_count, 3).mean(axis=1)
    structure_array = np.array([sample.index], dtype=np.int64)
    atom_count_array = np.array([atom_count], dtype=np.int64)
    component_indices = np.arange(component_count, dtype=np.int64)
    return {
        "structure_index": structure_array,
        "num_atoms": atom_count_array,
        "energy_pred_total": np.array([energy_pred_total]),
        "energy_true_total": np.array([energy_true_total]),
        "energy_pred_per_atom": np.array([energy_pred_per_atom]),
        "energy_true_per_atom": np.array([energy_true_per_atom]),
        "energy_residual": np.array([energy_pred_per_atom - energy_true_per_atom]),
        "energy_raw_var": energy_raw,
        "energy_calibrated_var": energy_var,
        "energy_calibrated_std": np.sqrt(energy_var),
        "energy_inverse_variance": 1.0 / energy_var,
        "energy_raw_var_total_derived": atom_count**2 * energy_raw,
        "energy_calibrated_var_total_derived": atom_count**2 * energy_var,
        "force_offsets": np.array([0, component_count], dtype=np.int64),
        "force_structure_index": np.full(component_count, sample.index, dtype=np.int64),
        "force_component_index_within_structure": component_indices,
        "force_atom_index": component_indices // 3,
        "force_cartesian_index": component_indices % 3,
        "force_pred": force_pred,
        "force_true": force_true,
        "force_residual": force_pred - force_true,
        "force_raw_var_component": force_raw,
        "force_calibrated_var_component": force_var,
        "force_calibrated_std_component": np.sqrt(force_var),
        "force_inverse_variance_component": 1.0 / force_var,
        "force_raw_var_atom_mean": force_atom_raw,
        "force_calibrated_var_atom_mean": force_atom_var,
        "force_calibrated_std_atom_rms": np.sqrt(force_atom_var),
        "force_raw_var_component_mean_structure": np.array([float(force_raw.mean())]),
        "force_calibrated_var_component_mean_structure": np.array(
            [float(force_var.mean())]
        ),
        "force_calibrated_std_component_rms_structure": np.array(
            [float(np.sqrt(force_var.mean()))]
        ),
    }


def merge_structure_results(
    results: Sequence[Mapping[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Concatenate complete results and rebuild global force offsets."""
    if not results:
        raise ValueError("cannot merge an empty result list")
    keys = set(results[0])
    if any(set(result) != keys for result in results):
        raise ValueError("evaluation result fields do not match")
    merged: dict[str, np.ndarray] = {}
    for key in sorted(keys - {"force_offsets"}):
        merged[key] = np.concatenate([np.atleast_1d(result[key]) for result in results])
    component_counts = [
        int(np.asarray(result["force_offsets"])[-1]) for result in results
    ]
    merged["force_offsets"] = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.cumsum(component_counts, dtype=np.int64),
        )
    )
    return merged


def summarize_evaluation(
    details: Mapping[str, np.ndarray],
) -> dict[str, object]:
    energy_residual = np.asarray(details["energy_residual"], dtype=np.float64)
    force_residual = np.asarray(details["force_residual"], dtype=np.float64)
    return {
        "structure_count": int(len(details["structure_index"])),
        "atom_count": int(np.sum(details["num_atoms"])),
        "force_component_count": int(len(force_residual)),
        "energy_rmse_per_atom": float(np.sqrt(np.mean(np.square(energy_residual)))),
        "force_rmse_component": float(np.sqrt(np.mean(np.square(force_residual)))),
        "energy_mean_calibrated_std": float(np.mean(details["energy_calibrated_std"])),
        "force_mean_calibrated_std_component": float(
            np.mean(details["force_calibrated_std_component"])
        ),
    }


def _load_calibration(path: Path) -> dict[str, CalibrationRecord]:
    value = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    allowed = {field.name for field in fields(CalibrationRecord)}
    return {
        target: CalibrationRecord(
            **{
                key: item
                for key, item in value["selected"][target].items()
                if key in allowed
            }
        )
        for target in ("energy", "force")
    }


def _flush_shard(
    shards_dir: Path,
    index: int,
    batch: list[Mapping[str, np.ndarray]],
) -> tuple[Path, dict[str, np.ndarray]]:
    arrays = merge_structure_results(batch)
    path = shards_dir / f"shard-{index:06d}.npz"
    atomic_npz_save(path, arrays)
    return path, arrays


def run_evaluate(config: LLPRConfig) -> Path:
    """Evaluate the configured test dataset with resumable complete shards."""
    curvature_dir = run_build(config)
    calibration_dir = run_calibrate(config)
    curvature_manifest = load_complete_manifest(curvature_dir / "manifest.json")
    calibration_manifest = load_complete_manifest(calibration_dir / "manifest.json")
    test = dataset_identity(config.data.test)
    if (
        config.data.test_expected_sha256 is not None
        and test.sha256 != config.data.test_expected_sha256
    ):
        raise ValueError("test dataset SHA mismatch")
    identity = stage_identity(
        "evaluation",
        {
            "curvature_identity": curvature_manifest["identity"],
            "calibration_identity": calibration_manifest["identity"],
            "test_sha256": test.sha256,
            "shard_structure_count": config.output.shard_structure_count,
        },
    )
    identity_value = str(identity["identity"])
    root = config.output.root / config.output.experiment
    stage_dir = (
        RunPaths(root).evaluation
        / str(calibration_manifest["identity"])
        / identity_value
    )
    manifest_path = stage_dir / "manifest.json"
    if manifest_path.exists():
        load_complete_manifest(manifest_path, {"identity": identity_value})
        return stage_dir
    stage_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = stage_dir / "shards"
    shards_dir.mkdir(exist_ok=True)
    progress_path = stage_dir / "progress.json"

    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress["identity"] != identity_value:
            raise ValueError("evaluation identity mismatch in progress")
        shard_records = list(progress["shards"])
        for record in shard_records:
            path = stage_dir / record["path"]
            if sha256_file(path) != record["sha256"]:
                raise ValueError(f"evaluation shard hash mismatch: {path}")
        next_index = int(progress["next_structure_index"])
    else:
        shard_records = []
        next_index = 0

    with np.load(curvature_dir / "curvature.npz", allow_pickle=False) as archive:
        matrices = {
            "energy": torch.from_numpy(archive["energy"].copy()).to(torch.float64),
            "force": torch.from_numpy(archive["force"].copy()).to(torch.float64),
        }
    calibration = _load_calibration(calibration_dir)
    solvers = {
        target: torch.linalg.cholesky(
            matrices[target]
            + calibration[target].eta
            * torch.eye(matrices[target].shape[0], dtype=torch.float64)
        )
        for target in ("energy", "force")
    }
    device = torch.device(config.runtime.device)
    loaded = load_checkpoint(config.checkpoint, device=device, dtype=torch.float64)
    layout = discover_readout_layout(loaded.model)
    batch: list[Mapping[str, np.ndarray]] = []
    for sample in iter_samples(config.data.test):
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
        batch.append(evaluate_structure(sample, jacobians, calibration, solvers))
        if len(batch) == config.output.shard_structure_count:
            shard_path, arrays = _flush_shard(shards_dir, len(shard_records), batch)
            shard_records.append(
                {
                    "path": str(shard_path.relative_to(stage_dir)),
                    "sha256": sha256_file(shard_path),
                    "structure_count": len(arrays["structure_index"]),
                    "force_component_count": len(arrays["force_residual"]),
                }
            )
            next_index = sample.index + 1
            atomic_json_dump(
                progress_path,
                {
                    "identity": identity_value,
                    "next_structure_index": next_index,
                    "shards": shard_records,
                },
            )
            batch = []
    if batch:
        shard_path, arrays = _flush_shard(shards_dir, len(shard_records), batch)
        shard_records.append(
            {
                "path": str(shard_path.relative_to(stage_dir)),
                "sha256": sha256_file(shard_path),
                "structure_count": len(arrays["structure_index"]),
                "force_component_count": len(arrays["force_residual"]),
            }
        )
        next_index = int(arrays["structure_index"][-1]) + 1
        atomic_json_dump(
            progress_path,
            {
                "identity": identity_value,
                "next_structure_index": next_index,
                "shards": shard_records,
            },
        )

    shard_arrays: list[dict[str, np.ndarray]] = []
    for record in shard_records:
        with np.load(stage_dir / record["path"], allow_pickle=False) as archive:
            shard_arrays.append({name: archive[name].copy() for name in archive.files})
    details = merge_structure_results(shard_arrays)
    details_path = stage_dir / "details.npz"
    summary_path = stage_dir / "summary.json"
    preview_path = stage_dir / "preview.json"
    atomic_npz_save(details_path, details)
    summary = summarize_evaluation(details)
    atomic_json_dump(summary_path, summary)
    atomic_json_dump(
        preview_path,
        {
            "summary": summary,
            "first_structure_indices": details["structure_index"][:10].tolist(),
        },
    )
    atomic_json_dump(
        manifest_path,
        {
            **identity,
            "status": "complete",
            "origin": "recomputed",
            "curvature_identity": curvature_manifest["identity"],
            "calibration_identity": calibration_manifest["identity"],
            "files": {
                details_path.name: sha256_file(details_path),
                summary_path.name: sha256_file(summary_path),
                preview_path.name: sha256_file(preview_path),
            },
        },
    )
    progress_path.unlink(missing_ok=True)
    if not config.output.keep_shards:
        shutil.rmtree(shards_dir)
    return stage_dir
