"""Publication path for evaluations with exactly one active target."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from .artifacts import atomic_torch_save, atomic_write_json, sha256_file
from .binning import BinningSpec, expected_error, labels_from_thresholds
from .config import ConfidenceConfig
from .errors import energy_per_atom_error, force_error, force_error_definition
from .metrics import classification_metrics
from .model import ConfidenceModel


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
    force_active = model.force_active
    energy_active = model.energy_active
    if force_active == energy_active:
        raise ValueError("single-target evaluation requires exactly one active target")
    names = ["structure_ids"]
    if force_active:
        names.extend(
            (
                "force_logits",
                "force_labels",
                "force_observed_errors",
                "force_expected_errors",
            )
        )
    else:
        names.extend(
            (
                "energy_logits",
                "energy_labels",
                "energy_observed_errors",
                "energy_expected_errors",
            )
        )
    collected: dict[str, list[torch.Tensor]] = {name: [] for name in names}
    atom_counts: list[torch.Tensor] = []
    with torch.inference_mode():
        for raw_batch in loader:
            batch = to_device(raw_batch, device)
            output = model(
                batch["force_features"] if force_active else None,
                batch["energy_features"] if energy_active else None,
                batch["atom_offsets"] if energy_active else None,
            )
            values: dict[str, torch.Tensor] = {"structure_ids": batch["structure_ids"]}
            if force_active:
                if output.force_logits is None:
                    raise RuntimeError("active force model returned no logits")
                observed = force_error(
                    batch["force_prediction"],
                    batch["force_reference"],
                    config.model.force.target_mode,
                )
                values.update(
                    {
                        "force_logits": output.force_logits,
                        "force_labels": labels_from_thresholds(
                            observed, force_spec.thresholds
                        ),
                        "force_observed_errors": observed,
                        "force_expected_errors": expected_error(
                            output.force_logits, force_spec.representatives
                        ),
                    }
                )
            else:
                if output.energy_logits is None:
                    raise RuntimeError("active energy model returned no logits")
                observed = energy_per_atom_error(
                    batch["energy_prediction"],
                    batch["energy_reference"],
                    batch["num_atoms"],
                )
                values.update(
                    {
                        "energy_logits": output.energy_logits,
                        "energy_labels": labels_from_thresholds(
                            observed, energy_spec.thresholds
                        ),
                        "energy_observed_errors": observed,
                        "energy_expected_errors": expected_error(
                            output.energy_logits, energy_spec.representatives
                        ),
                    }
                )
            for name, tensor in values.items():
                collected[name].append(tensor.detach().cpu())
            atom_counts.append(batch["num_atoms"].detach().cpu())

    predictions = {name: torch.cat(parts) for name, parts in collected.items()}
    predictions["atom_offsets"] = offsets(atom_counts)
    if force_active:
        predictions["force_representatives"] = force_spec.representatives
        predictions["force_target_mode"] = config.model.force.target_mode
        predictions["force_error_definition"] = force_error_definition(
            config.model.force.target_mode
        )
        prefix = "force"
        spec = force_spec
    else:
        predictions["energy_representatives"] = energy_spec.representatives
        prefix = "energy"
        spec = energy_spec

    prediction_path = evaluation_dir / "test_predictions.pt"
    assert_write()
    atomic_torch_save(prediction_path, predictions)
    logits = predictions[f"{prefix}_logits"]
    labels = predictions[f"{prefix}_labels"]
    observed = predictions[f"{prefix}_observed_errors"]
    expected = predictions[f"{prefix}_expected_errors"]
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    flat_observed = observed.reshape(-1)
    flat_expected = expected.reshape(-1)
    metrics = {
        prefix: classification_metrics(
            flat_logits,
            flat_labels,
            flat_observed,
            spec.representatives,
        )
    }
    metrics_path = evaluation_dir / "metrics.json"
    assert_write()
    atomic_write_json(metrics_path, metrics)
    csv_path = evaluation_dir / f"{prefix}_bin_summary.csv"
    assert_write()
    write_csv(csv_path, flat_labels, flat_observed, flat_expected, spec.num_bins)
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
    if force_active:
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
    if force_active:
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
