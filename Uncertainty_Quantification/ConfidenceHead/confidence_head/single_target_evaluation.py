"""Publication path for evaluations with exactly one active target."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from .artifacts import atomic_torch_save, atomic_write_json, sha256_file
from .binning import BinningSpec
from .config import ConfidenceConfig
from .errors import force_error_definition
from .model import ConfidenceModel
from .single_target_prediction import collect_single_target_predictions


def evaluate_single_target(
    *,
    model: ConfidenceModel,
    loader: Any,
    device: torch.device,
    config: ConfidenceConfig,
    force_spec: BinningSpec,
    energy_spec: BinningSpec,
    evaluation_dir: Path,
    run_dir: Path,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    cache_identity: str,
    checkpoint_relative: str,
    checkpoint_sha: str,
    started_at: str,
    evaluation_schema_version: str,
    to_device: Callable[[Mapping[str, Any], torch.device], dict[str, Any]],
    offsets: Callable[[list[torch.Tensor]], torch.Tensor],
    write_csv: Callable[[Path, torch.Tensor, torch.Tensor, torch.Tensor, int], None],
    assert_write: Callable[[], None],
) -> Path:
    result = collect_single_target_predictions(
        model=model,
        loader=loader,
        device=device,
        config=config,
        force_spec=force_spec,
        energy_spec=energy_spec,
        to_device=to_device,
        offsets=offsets,
        include_raw=False,
    )
    prefix = result.target
    predictions = result.predictions
    spec = force_spec if prefix == "force" else energy_spec

    prediction_path = evaluation_dir / "test_predictions.pt"
    assert_write()
    atomic_torch_save(prediction_path, predictions)
    metrics_path = evaluation_dir / "metrics.json"
    assert_write()
    atomic_write_json(metrics_path, result.metrics)
    csv_path = evaluation_dir / f"{prefix}_bin_summary.csv"
    assert_write()
    write_csv(
        csv_path,
        predictions[f"{prefix}_labels"].reshape(-1),
        predictions[f"{prefix}_observed_errors"].reshape(-1),
        predictions[f"{prefix}_expected_errors"].reshape(-1),
        spec.num_bins,
    )
    artifacts = {
        path.name: {"path": path.name, "sha256": sha256_file(path)}
        for path in (prediction_path, metrics_path, csv_path)
    }
    atom_count = int(predictions["atom_offsets"][-1])
    counts: dict[str, Any] = {
        "structures": len(predictions["structure_ids"]),
        "atoms": atom_count,
        "force_components": 3 * atom_count,
    }
    if prefix == "force":
        counts.update(
            {
                "force_targets": predictions["force_labels"].numel(),
                "force_target_mode": config.model.force.target_mode,
            }
        )
    evaluation_manifest = {
        "schema_version": evaluation_schema_version,
        "status": "complete",
        "identity": manifest["run_id"],
        "run_id": manifest["run_id"],
        "cache_id": cache_identity,
        "active_targets": [prefix],
        "checkpoint": {"path": checkpoint_relative, "sha256": checkpoint_sha},
        "test_counts": counts,
        "artifacts": artifacts,
        "started_at": started_at,
        "completed_at": datetime.now(UTC).isoformat(),
    }
    if prefix == "force":
        evaluation_manifest.update(
            {
                "force_target_mode": config.model.force.target_mode,
                "force_error_definition": force_error_definition(
                    config.model.force.target_mode
                ),
            }
        )
    evaluation_manifest_path = evaluation_dir / "manifest.json"
    assert_write()
    atomic_write_json(evaluation_manifest_path, evaluation_manifest)

    updated = dict(manifest)
    declared = dict(updated["artifacts"])
    for name, entry in artifacts.items():
        declared[f"evaluation/{name}"] = {
            "path": f"evaluation/{name}",
            "sha256": entry["sha256"],
        }
    declared["evaluation/manifest.json"] = {
        "path": "evaluation/manifest.json",
        "sha256": sha256_file(evaluation_manifest_path),
    }
    updated["artifacts"] = declared
    updated["evaluation"] = {
        "status": "complete",
        "manifest": "evaluation/manifest.json",
    }
    assert_write()
    atomic_write_json(manifest_path, updated)
    assert_write()
    return evaluation_dir
