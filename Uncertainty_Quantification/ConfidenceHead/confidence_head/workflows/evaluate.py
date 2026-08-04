"""Evaluate the best confidence-head checkpoint without creating figures."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml

from ..artifacts import (
    atomic_torch_save,
    atomic_write_json,
    load_verified_torch,
    sha256_file,
)
from ..binning import (
    BinningSpec,
    expected_error,
    fixed_linear_binning,
    labels_from_thresholds,
)
from ..cache import SCHEMA_VERSION as CACHE_SCHEMA_VERSION
from ..cache import CachedSplitDataset
from ..config import ConfidenceConfig, ForceTargetMode
from ..errors import energy_per_atom_error, force_error, force_error_definition
from ..identity import cache_id
from ..metrics import classification_metrics
from ..model import ConfidenceModel
from ..single_target_evaluation import evaluate_single_target
from ..trainer import CHECKPOINT_SCHEMA_VERSION
from .train import (
    RUN_SCHEMA_VERSION,
    _assert_run_directory_identity,
    _bin_payload,
    _exclusive_run_lock,
    _identities,
    _loader,
    _run_directory_identity,
    _safe_run_dir,
    _to_device,
)


EVALUATION_SCHEMA_VERSION = "upet_confidence_evaluation_v1"


def _load_mapping(path: Path, *, yaml_file: bool = False) -> dict[str, Any]:
    try:
        text = Path(path).read_text(encoding="utf-8")
        value = yaml.safe_load(text) if yaml_file else json.loads(text)
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as error:
        raise ValueError(f"invalid artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"artifact {path} must contain a mapping")
    return value


def _specs(config: ConfidenceConfig) -> tuple[BinningSpec, BinningSpec]:
    return (
        fixed_linear_binning(
            config.model.force.num_bins, config.binning.force_max_error
        ),
        fixed_linear_binning(
            config.model.energy.num_bins, config.binning.energy_max_error
        ),
    )


def _write_csv(
    path: Path,
    labels: torch.Tensor,
    observed: torch.Tensor,
    expected: torch.Tensor,
    bins: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "bin",
                    "sample_count",
                    "mean_observed_error",
                    "mean_expected_error",
                ),
            )
            writer.writeheader()
            for index in range(bins):
                mask = labels == index
                count = int(mask.sum().item())
                writer.writerow(
                    {
                        "bin": index,
                        "sample_count": count,
                        "mean_observed_error": float(observed[mask].mean())
                        if count
                        else "",
                        "mean_expected_error": float(expected[mask].mean())
                        if count
                        else "",
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _offsets(counts: list[torch.Tensor]) -> torch.Tensor:
    if not counts:
        raise ValueError("test split produced no batches")
    merged = torch.cat(counts).to(torch.int64)
    result = torch.zeros(len(merged) + 1, dtype=torch.int64)
    result[1:] = torch.cumsum(merged, dim=0)
    return result


def _evaluation_directory_identity(
    run_dir: Path,
) -> tuple[Path, tuple[int, int]]:
    evaluation_dir = run_dir / "evaluation"
    if evaluation_dir.is_symlink():
        raise ValueError("evaluation directory must not be a symlink")
    evaluation_dir.mkdir(exist_ok=True)
    if evaluation_dir.is_symlink() or not evaluation_dir.is_dir():
        raise ValueError("evaluation directory must be a regular directory")
    resolved = evaluation_dir.resolve()
    if not resolved.is_relative_to(run_dir.resolve()):
        raise ValueError("evaluation directory escapes run directory")
    stat = evaluation_dir.stat(follow_symlinks=False)
    return evaluation_dir, (stat.st_dev, stat.st_ino)


def _assert_evaluation_directory_identity(
    run_dir: Path,
    evaluation_dir: Path,
    expected: tuple[int, int],
) -> None:
    if evaluation_dir.is_symlink():
        raise ValueError("evaluation directory must not be a symlink")
    resolved = evaluation_dir.resolve()
    if not resolved.is_relative_to(run_dir.resolve()):
        raise ValueError("evaluation directory escapes run directory")
    stat = evaluation_dir.stat(follow_symlinks=False)
    if (stat.st_dev, stat.st_ino) != expected:
        raise ValueError("evaluation directory identity changed during publication")


def _assert_evaluation_write_target(
    *,
    run_dir: Path,
    output_root: Path,
    run_identity: tuple[int, int],
    evaluation_dir: Path,
    evaluation_identity: tuple[int, int],
) -> None:
    _assert_run_directory_identity(run_dir, output_root, run_identity)
    _assert_evaluation_directory_identity(
        run_dir,
        evaluation_dir,
        evaluation_identity,
    )


def _declared_artifact(run_dir: Path, manifest: Mapping[str, Any], path: Path) -> Path:
    candidate = path.resolve()
    if not candidate.is_relative_to(run_dir):
        raise ValueError("checkpoint escapes run directory")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("run artifacts declaration is missing")
    for entry in artifacts.values():
        if not isinstance(entry, Mapping):
            continue
        declared = (run_dir / str(entry.get("path"))).resolve()
        if declared != candidate:
            continue
        if not candidate.is_file():
            raise ValueError("declared checkpoint is missing")
        expected = entry.get("sha256")
        if not isinstance(expected, str) or sha256_file(candidate) != expected:
            raise ValueError("declared checkpoint sha256 mismatch")
        return candidate
    raise ValueError("checkpoint is not declared by run manifest")


def _validate_force_semantics(
    payload: Mapping[str, Any], expected_mode: ForceTargetMode, *, context: str
) -> None:
    actual_mode = payload.get("force_target_mode")
    actual_definition = payload.get("force_error_definition")
    if (
        actual_mode is None
        and actual_definition is None
        and expected_mode == "component"
    ):
        return
    if actual_mode != expected_mode or actual_definition != force_error_definition(
        expected_mode
    ):
        raise ValueError(f"{context} force target semantics mismatch")


def _checkpoint_identity(
    snapshot: Mapping[str, Any],
    manifest: Mapping[str, Any],
    expected_mode: ForceTargetMode,
) -> None:
    if snapshot.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("checkpoint schema mismatch")
    for field in ("config_id", "cache_id", "binning_id", "model_loss_id"):
        if snapshot.get(field) != manifest.get(field):
            raise ValueError(f"checkpoint {field} mismatch")
    _validate_force_semantics(manifest, expected_mode, context="run manifest")
    _validate_force_semantics(snapshot, expected_mode, context="checkpoint")


def _evaluate_run_locked(
    run_dir: Path,
    *,
    output_root: Path,
    run_identity: tuple[int, int],
    cache_manifest_path: Path,
    checkpoint_path: Path | None = None,
) -> Path:
    """Evaluate one checkpoint, defaulting to the run's best checkpoint."""
    started_at = datetime.now(UTC).isoformat()
    run_dir = Path(run_dir).resolve()
    manifest_path = run_dir / "manifest.json"
    manifest = _load_mapping(manifest_path)
    if (
        manifest.get("schema_version") != RUN_SCHEMA_VERSION
        or manifest.get("status") != "complete"
    ):
        raise ValueError("run manifest must be complete")
    for relative in ("resolved_config.yaml", "binning.json"):
        _declared_artifact(run_dir, manifest, run_dir / relative)
    config = ConfidenceConfig.model_validate(
        _load_mapping(run_dir / "resolved_config.yaml", yaml_file=True)
    )
    force_spec, energy_spec = _specs(config)
    bins = _load_mapping(run_dir / "binning.json")
    if bins != _bin_payload(force_spec, energy_spec, config.model.force.target_mode):
        raise ValueError("binning artifact identity disagrees with resolved config")
    cache_path = Path(cache_manifest_path).resolve()
    cache_manifest = _load_mapping(cache_path)
    if cache_manifest.get("status") != "complete":
        raise ValueError("evaluation cache manifest must be complete")
    cache_payload = cache_manifest.get("identity_payload")
    _validate_force_semantics(
        manifest, config.model.force.target_mode, context="run manifest"
    )
    if not isinstance(cache_payload, Mapping):
        raise ValueError("evaluation cache identity_payload is missing")
    derived_cache_id = cache_id(
        {
            "schema_version": CACHE_SCHEMA_VERSION,
            "identity_payload": dict(cache_payload),
        }
    )
    if (
        cache_manifest.get("schema_version") != CACHE_SCHEMA_VERSION
        or cache_manifest.get("identity") != derived_cache_id
        or cache_manifest.get("cache_id") != derived_cache_id
        or manifest.get("cache_id") != derived_cache_id
    ):
        raise ValueError("evaluation cache identity mismatch")
    cache_identity = derived_cache_id
    identity, expected_run_id, _ = _identities(config, cache_manifest, bins)
    for field, expected in (
        ("config_id", identity.config_id),
        ("cache_id", identity.cache_id),
        ("binning_id", identity.binning_id),
        ("model_loss_id", identity.model_loss_id),
        ("run_id", expected_run_id),
        ("identity", expected_run_id),
    ):
        if manifest.get(field) != expected:
            raise ValueError(f"evaluation run {field} identity mismatch")
    evaluation_dir, evaluation_identity = _evaluation_directory_identity(run_dir)
    _assert_evaluation_write_target(
        run_dir=run_dir,
        output_root=output_root,
        run_identity=run_identity,
        evaluation_dir=evaluation_dir,
        evaluation_identity=evaluation_identity,
    )
    dataset = CachedSplitDataset(cache_path, "test", cache_identity)

    if checkpoint_path is None:
        checkpoint = run_dir / "checkpoints" / "best.pt"
        checkpoint_relative = "../checkpoints/best.pt"
    else:
        checkpoint = Path(checkpoint_path).resolve()
        checkpoint_relative = os.path.relpath(checkpoint, run_dir / "evaluation")
    if not checkpoint.is_file():
        raise ValueError(f"evaluation checkpoint does not exist: {checkpoint}")
    checkpoint = _declared_artifact(run_dir, manifest, checkpoint)
    checkpoint_sha = next(
        str(entry["sha256"])
        for entry in manifest["artifacts"].values()
        if isinstance(entry, Mapping)
        and (run_dir / str(entry.get("path"))).resolve() == checkpoint
    )
    snapshot = load_verified_torch(
        checkpoint,
        expected_sha256=checkpoint_sha,
        weights_only=False,
    )
    if not isinstance(snapshot, Mapping):
        raise ValueError("checkpoint must contain a mapping")
    _checkpoint_identity(snapshot, manifest, config.model.force.target_mode)

    model = ConfidenceModel(
        force_input_dim=int(manifest["force_feature_dim"]),
        energy_input_dim=int(manifest["energy_feature_dim"]),
        force_hidden_dims=config.model.force.hidden_dims,
        energy_hidden_dims=config.model.energy.hidden_dims,
        force_dropout=config.model.force.dropout,
        energy_dropout=config.model.energy.dropout,
        force_num_bins=config.model.force.num_bins,
        energy_num_bins=config.model.energy.num_bins,
        cumulant_order=config.model.energy.cumulant_order,
        signed_root=config.model.energy.signed_root,
        force_target_mode=config.model.force.target_mode,
        force_active=config.model.force.enabled and config.loss.force_coefficient > 0,
        energy_active=config.model.energy.enabled
        and config.loss.energy_coefficient > 0,
    )
    model.load_state_dict(snapshot["model"])
    device = torch.device(config.run.device)
    model.to(device).eval()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.run.seed)
    loader = _loader(dataset, config, shuffle=False, generator=generator)

    if model.force_active != model.energy_active:

        def assert_write() -> None:
            _assert_evaluation_write_target(
                run_dir=run_dir,
                output_root=output_root,
                run_identity=run_identity,
                evaluation_dir=evaluation_dir,
                evaluation_identity=evaluation_identity,
            )

        return evaluate_single_target(
            model=model,
            loader=loader,
            device=device,
            config=config,
            force_spec=force_spec,
            energy_spec=energy_spec,
            evaluation_dir=evaluation_dir,
            run_dir=run_dir,
            manifest=manifest,
            manifest_path=manifest_path,
            cache_identity=cache_identity,
            checkpoint_relative=checkpoint_relative,
            checkpoint_sha=checkpoint_sha,
            started_at=started_at,
            evaluation_schema_version=EVALUATION_SCHEMA_VERSION,
            to_device=_to_device,
            offsets=_offsets,
            write_csv=_write_csv,
            assert_write=assert_write,
        )

    collected: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "force_logits",
            "force_labels",
            "force_observed_errors",
            "force_expected_errors",
            "energy_logits",
            "energy_labels",
            "energy_observed_errors",
            "energy_expected_errors",
            "structure_ids",
        )
    }
    atom_counts: list[torch.Tensor] = []
    with torch.inference_mode():
        for raw_batch in loader:
            batch = _to_device(raw_batch, device)
            output = model(
                batch["force_features"],
                batch["energy_features"],
                batch["atom_offsets"],
            )
            force_observed = force_error(
                batch["force_prediction"],
                batch["force_reference"],
                config.model.force.target_mode,
            )
            energy_observed = energy_per_atom_error(
                batch["energy_prediction"],
                batch["energy_reference"],
                batch["num_atoms"],
            )
            values = {
                "force_logits": output.force_logits,
                "force_labels": labels_from_thresholds(
                    force_observed, force_spec.thresholds
                ),
                "force_observed_errors": force_observed,
                "force_expected_errors": expected_error(
                    output.force_logits, force_spec.representatives
                ),
                "energy_logits": output.energy_logits,
                "energy_labels": labels_from_thresholds(
                    energy_observed, energy_spec.thresholds
                ),
                "energy_observed_errors": energy_observed,
                "energy_expected_errors": expected_error(
                    output.energy_logits, energy_spec.representatives
                ),
                "structure_ids": batch["structure_ids"],
            }
            for name, tensor in values.items():
                collected[name].append(tensor.detach().cpu())
            atom_counts.append(batch["num_atoms"].detach().cpu())

    predictions = {name: torch.cat(parts, dim=0) for name, parts in collected.items()}
    predictions["atom_offsets"] = _offsets(atom_counts)
    predictions["force_representatives"] = force_spec.representatives
    predictions["energy_representatives"] = energy_spec.representatives
    predictions["force_target_mode"] = config.model.force.target_mode
    predictions["force_error_definition"] = force_error_definition(
        config.model.force.target_mode
    )
    prediction_path = evaluation_dir / "test_predictions.pt"
    _assert_evaluation_write_target(
        run_dir=run_dir,
        output_root=output_root,
        run_identity=run_identity,
        evaluation_dir=evaluation_dir,
        evaluation_identity=evaluation_identity,
    )
    atomic_torch_save(prediction_path, predictions)
    _assert_evaluation_write_target(
        run_dir=run_dir,
        output_root=output_root,
        run_identity=run_identity,
        evaluation_dir=evaluation_dir,
        evaluation_identity=evaluation_identity,
    )

    force_logits = predictions["force_logits"].reshape(
        -1, predictions["force_logits"].shape[-1]
    )
    force_labels = predictions["force_labels"].reshape(-1)
    force_observed = predictions["force_observed_errors"].reshape(-1)
    force_expected = predictions["force_expected_errors"].reshape(-1)
    energy_logits = predictions["energy_logits"]
    energy_labels = predictions["energy_labels"]
    energy_observed = predictions["energy_observed_errors"]
    energy_expected = predictions["energy_expected_errors"]
    metrics = {
        "force": classification_metrics(
            force_logits, force_labels, force_observed, force_spec.representatives
        ),
        "energy": classification_metrics(
            energy_logits, energy_labels, energy_observed, energy_spec.representatives
        ),
    }
    metrics_path = evaluation_dir / "metrics.json"
    _assert_evaluation_write_target(
        run_dir=run_dir,
        output_root=output_root,
        run_identity=run_identity,
        evaluation_dir=evaluation_dir,
        evaluation_identity=evaluation_identity,
    )
    atomic_write_json(metrics_path, metrics)
    _assert_evaluation_write_target(
        run_dir=run_dir,
        output_root=output_root,
        run_identity=run_identity,
        evaluation_dir=evaluation_dir,
        evaluation_identity=evaluation_identity,
    )
    force_csv = evaluation_dir / "force_bin_summary.csv"
    energy_csv = evaluation_dir / "energy_bin_summary.csv"
    _assert_evaluation_write_target(
        run_dir=run_dir,
        output_root=output_root,
        run_identity=run_identity,
        evaluation_dir=evaluation_dir,
        evaluation_identity=evaluation_identity,
    )
    _write_csv(
        force_csv, force_labels, force_observed, force_expected, force_spec.num_bins
    )
    _assert_evaluation_write_target(
        run_dir=run_dir,
        output_root=output_root,
        run_identity=run_identity,
        evaluation_dir=evaluation_dir,
        evaluation_identity=evaluation_identity,
    )
    _write_csv(
        energy_csv,
        energy_labels,
        energy_observed,
        energy_expected,
        energy_spec.num_bins,
    )
    _assert_evaluation_write_target(
        run_dir=run_dir,
        output_root=output_root,
        run_identity=run_identity,
        evaluation_dir=evaluation_dir,
        evaluation_identity=evaluation_identity,
    )
    artifacts = {
        path.name: {"path": path.name, "sha256": sha256_file(path)}
        for path in (prediction_path, metrics_path, force_csv, energy_csv)
    }
    evaluation_manifest = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "complete",
        "identity": manifest["run_id"],
        "run_id": manifest["run_id"],
        "cache_id": cache_identity,
        "force_target_mode": config.model.force.target_mode,
        "force_error_definition": force_error_definition(
            config.model.force.target_mode
        ),
        "checkpoint": {
            "path": checkpoint_relative,
            "sha256": checkpoint_sha,
        },
        "test_counts": {
            "structures": len(predictions["structure_ids"]),
            "atoms": int(predictions["atom_offsets"][-1]),
            "force_components": 3 * int(predictions["atom_offsets"][-1]),
            "force_targets": predictions["force_labels"].numel(),
            "force_target_mode": config.model.force.target_mode,
        },
        "artifacts": artifacts,
        "started_at": started_at,
        "completed_at": datetime.now(UTC).isoformat(),
    }
    evaluation_manifest_path = evaluation_dir / "manifest.json"
    _assert_evaluation_write_target(
        run_dir=run_dir,
        output_root=output_root,
        run_identity=run_identity,
        evaluation_dir=evaluation_dir,
        evaluation_identity=evaluation_identity,
    )
    atomic_write_json(evaluation_manifest_path, evaluation_manifest)
    _assert_evaluation_write_target(
        run_dir=run_dir,
        output_root=output_root,
        run_identity=run_identity,
        evaluation_dir=evaluation_dir,
        evaluation_identity=evaluation_identity,
    )

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
    _assert_evaluation_write_target(
        run_dir=run_dir,
        output_root=output_root,
        run_identity=run_identity,
        evaluation_dir=evaluation_dir,
        evaluation_identity=evaluation_identity,
    )
    atomic_write_json(manifest_path, updated)
    _assert_evaluation_write_target(
        run_dir=run_dir,
        output_root=output_root,
        run_identity=run_identity,
        evaluation_dir=evaluation_dir,
        evaluation_identity=evaluation_identity,
    )
    return evaluation_dir


def evaluate_run(
    run_dir: Path,
    *,
    cache_manifest_path: Path,
    checkpoint_path: Path | None = None,
) -> Path:
    """Evaluate one run while serializing and confining all publications."""
    requested = Path(run_dir)
    if requested.is_symlink() or requested.parent.name != "runs":
        raise ValueError("evaluation run directory must be a regular derived run")
    output_root = requested.parent.parent
    run_name = requested.name
    with _exclusive_run_lock(output_root, run_name):
        safe_run_dir = _safe_run_dir(output_root, run_name, resume=True)
        if safe_run_dir.resolve() != requested.resolve():
            raise ValueError("evaluation run directory does not match derived run")
        run_identity = _run_directory_identity(safe_run_dir, output_root)
        result = _evaluate_run_locked(
            safe_run_dir,
            output_root=output_root,
            run_identity=run_identity,
            cache_manifest_path=cache_manifest_path,
            checkpoint_path=checkpoint_path,
        )
        _assert_run_directory_identity(safe_run_dir, output_root, run_identity)
        return result
