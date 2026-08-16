"""Dataset-scoped ConfidenceHead prediction and UQ publication."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from .artifacts import (
    atomic_torch_save,
    atomic_write_json,
    load_verified_torch,
    sha256_file,
)
from .binning import BinningSpec
from .cache import collate_cached_structures
from .config import ConfidenceConfig
from .external_cache import (
    DatasetCache,
    build_external_dataset_cache,
    resolve_declared_cache_split,
)
from .external_config import (
    CacheSplitSource,
    ExistingEvaluationSource,
    ExternalPredictionConfig,
    ExtXYZSource,
)
from .model import ConfidenceModel
from .single_target_prediction import (
    SingleTargetResult,
    collect_single_target_predictions,
)
from .workflows.evaluate import (
    _checkpoint_identity,
    _declared_artifact,
    _offsets,
    _specs,
    _write_csv,
)
from .workflows.train import _to_device
from .workflows.verify import verify_run


EXTERNAL_PREDICTION_SCHEMA_VERSION = "upet_confidence_external_prediction_v1"


@dataclass(frozen=True)
class ExternalRuns:
    """The exact eight energy runs and one force-only run."""

    force: Path
    energy_by_order: Mapping[int, Path]

    def ordered(self) -> tuple[Path, ...]:
        return tuple(self.energy_by_order[index] for index in range(1, 9)) + (
            self.force,
        )


def _json_mapping(path: Path, *, context: str) -> dict[str, Any]:
    try:
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {context} {path}: {error}") from error
    if not isinstance(loaded, dict):
        raise ValueError(f"{context} must contain a mapping: {path}")
    return loaded


def _yaml_mapping(path: Path, *, context: str) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"invalid {context} {path}: {error}") from error
    if not isinstance(loaded, dict):
        raise ValueError(f"{context} must contain a mapping: {path}")
    return loaded


def _number(mapping: Mapping[str, Any], key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"resolved config field {key} must be numeric")
    return float(value)


def discover_external_runs(runs_root: Path) -> ExternalRuns:
    """Discover exactly one force-only run and energy orders one through eight."""

    root = Path(runs_root).resolve()
    if root.name != "runs" or not root.is_dir():
        raise ValueError("runs_root must be an existing runs directory")
    force: Path | None = None
    energies: dict[int, Path] = {}
    for run in sorted(path for path in root.iterdir() if path.is_dir()):
        manifest_path = run / "manifest.json"
        config_path = run / "resolved_config.yaml"
        if not manifest_path.is_file() or not config_path.is_file():
            continue
        manifest = _json_mapping(manifest_path, context="run manifest")
        if manifest.get("status") != "complete":
            continue
        resolved = _yaml_mapping(config_path, context="resolved config")
        loss = resolved.get("loss")
        model = resolved.get("model")
        if not isinstance(loss, Mapping) or not isinstance(model, Mapping):
            raise ValueError(f"run {run.name}: target configuration is missing")
        force_weight = _number(loss, "force_coefficient")
        energy_weight = _number(loss, "energy_coefficient")
        if force_weight > 0 and energy_weight == 0:
            force_config = model.get("force")
            if (
                not isinstance(force_config, Mapping)
                or force_config.get("target_mode") != "atom_mean"
            ):
                raise ValueError(f"run {run.name}: force target must be atom_mean")
            if force is not None:
                raise ValueError("duplicate force-only run")
            force = run
        elif force_weight == 0 and energy_weight > 0:
            energy_config = model.get("energy")
            order = (
                energy_config.get("cumulant_order")
                if isinstance(energy_config, Mapping)
                else None
            )
            if not isinstance(order, int) or isinstance(order, bool):
                raise ValueError(f"run {run.name}: invalid energy order")
            if order in energies:
                raise ValueError(f"duplicate energy order {order}")
            energies[order] = run
    if force is None:
        raise ValueError("force-only run was not found")
    if set(energies) != set(range(1, 9)):
        raise ValueError("energy runs must cover orders 1 through 8 exactly")
    return ExternalRuns(force=force, energy_by_order=energies)


def _identity(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return f"prediction-{hashlib.sha256(encoded).hexdigest()[:16]}"


def _confined_artifact(root: Path, relative: object) -> Path:
    if not isinstance(relative, str):
        raise ValueError("prediction artifact path must be a relative string")
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ValueError("prediction artifact path must be relative")
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("prediction artifact escapes publication directory")
    return resolved


def verify_external_prediction(
    manifest_path: Path,
    *,
    full: bool = True,
) -> dict[str, Any]:
    """Verify one complete external prediction and all declared artifacts."""

    path = Path(manifest_path).resolve()
    manifest = _json_mapping(path, context="external prediction manifest")
    payload = manifest.get("identity_payload")
    if (
        manifest.get("schema_version") != EXTERNAL_PREDICTION_SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or not isinstance(payload, Mapping)
        or manifest.get("identity") != _identity(payload)
    ):
        raise ValueError("external prediction schema/status/identity mismatch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise ValueError("external prediction artifacts are missing")
    for name, descriptor in artifacts.items():
        if not isinstance(name, str) or not isinstance(descriptor, Mapping):
            raise ValueError("external prediction artifact descriptor is invalid")
        artifact = _confined_artifact(path.parent, descriptor.get("path"))
        if not artifact.is_file():
            raise ValueError(f"external prediction artifact is missing: {name}")
        expected = descriptor.get("sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"external prediction artifact SHA is invalid: {name}")
        if full and sha256_file(artifact) != expected:
            raise ValueError(f"external prediction artifact SHA mismatch: {name}")
    predictions_descriptor = artifacts.get("predictions.pt")
    if not isinstance(predictions_descriptor, Mapping):
        raise ValueError("external prediction tensor artifact is missing")
    predictions = load_verified_torch(
        _confined_artifact(path.parent, predictions_descriptor.get("path")),
        expected_sha256=str(predictions_descriptor.get("sha256")),
        weights_only=True,
    )
    if not isinstance(predictions, Mapping):
        raise ValueError("external predictions must contain a mapping")
    target = payload.get("target")
    required = {
        "structure_ids",
        "atom_offsets",
        f"{target}_prediction",
        f"{target}_reference",
        f"{target}_logits",
        f"{target}_labels",
        f"{target}_observed_errors",
        f"{target}_expected_errors",
        f"{target}_representatives",
    }
    if target == "force":
        required |= {"force_target_mode", "force_error_definition"}
    if set(predictions) != required:
        raise ValueError("external prediction fields mismatch")
    tensors = [
        value for value in predictions.values() if isinstance(value, torch.Tensor)
    ]
    if not tensors or not all(bool(torch.isfinite(value).all()) for value in tensors):
        raise ValueError("external predictions must contain finite tensors")
    return manifest


def _write_bin_csv(
    path: Path,
    result: SingleTargetResult,
    spec: BinningSpec,
) -> None:
    labels = result.predictions[f"{result.target}_labels"].reshape(-1)
    observed = result.predictions[f"{result.target}_observed_errors"].reshape(-1)
    expected = result.predictions[f"{result.target}_expected_errors"].reshape(-1)
    _write_csv(path, labels, observed, expected, spec.num_bins)


def _publish_external_result(
    *,
    run_dir: Path,
    dataset_name: str,
    result: SingleTargetResult,
    spec: BinningSpec,
    identity_payload: Mapping[str, Any],
) -> Path:
    """Atomically publish one active head without changing the source run."""

    run = Path(run_dir).resolve()
    if run.parent.name != "runs":
        raise ValueError("external prediction run must be below a runs directory")
    payload = dict(identity_payload)
    identity = _identity(payload)
    predictions_root = run / "predictions"
    target = predictions_root / dataset_name
    manifest_path = target / "manifest.json"
    if manifest_path.is_file():
        existing = verify_external_prediction(manifest_path, full=True)
        if existing.get("identity") != identity:
            raise ValueError(
                "external prediction identity conflicts with existing result"
            )
        return target
    if target.exists():
        raise ValueError(
            "external prediction target exists without a complete identity"
        )
    predictions_root.mkdir(exist_ok=True)
    staging = predictions_root / f".staging-{uuid.uuid4().hex}"
    staging.mkdir()
    started_at = datetime.now(UTC).isoformat()
    try:
        prediction_path = staging / "predictions.pt"
        metrics_path = staging / "metrics.json"
        csv_path = staging / f"{result.target}_bin_summary.csv"
        atomic_torch_save(prediction_path, result.predictions)
        atomic_write_json(metrics_path, result.metrics)
        _write_bin_csv(csv_path, result, spec)
        artifacts = {
            artifact.name: {
                "path": artifact.name,
                "sha256": sha256_file(artifact),
            }
            for artifact in (prediction_path, metrics_path, csv_path)
        }
        offsets = result.predictions["atom_offsets"]
        structures = result.predictions["structure_ids"]
        manifest = {
            "schema_version": EXTERNAL_PREDICTION_SCHEMA_VERSION,
            "status": "complete",
            "identity": identity,
            "identity_payload": payload,
            "dataset": payload["dataset"],
            "counts": {
                "structures": int(len(structures)),
                "atoms": int(offsets[-1]),
                "targets": int(result.predictions[f"{result.target}_labels"].numel()),
            },
            "artifacts": artifacts,
            "started_at": started_at,
            "completed_at": datetime.now(UTC).isoformat(),
        }
        atomic_write_json(staging / "manifest.json", manifest)
        verify_external_prediction(staging / "manifest.json", full=True)
        os.replace(staging, target)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return target


def _validate_external_compatibility(
    run_manifest: Mapping[str, Any],
    run_config: ConfidenceConfig,
    cache: DatasetCache,
) -> None:
    compatibility = cache.compatibility
    if compatibility.checkpoint_sha256 != run_config.checkpoint.expected_sha256:
        raise ValueError("external cache base checkpoint SHA mismatch")
    if dict(compatibility.readouts) != run_config.readouts.model_dump():
        raise ValueError("external cache readout compatibility mismatch")
    if compatibility.force_dim != run_manifest.get("force_feature_dim"):
        raise ValueError("external cache force feature dimension mismatch")
    if compatibility.energy_dim != run_manifest.get("energy_feature_dim"):
        raise ValueError("external cache energy feature dimension mismatch")


def _run_config(run_dir: Path) -> ConfidenceConfig:
    return ConfidenceConfig.model_validate(
        _yaml_mapping(run_dir / "resolved_config.yaml", context="resolved config")
    )


def predict_dataset_run(
    run_dir: Path,
    dataset_name: str,
    cache: DatasetCache,
    *,
    batch_size: int,
    device_name: str,
) -> Path:
    """Run the verified best head checkpoint on one compatible raw cache."""

    run = Path(run_dir).resolve()
    run_manifest = _json_mapping(run / "manifest.json", context="run manifest")
    if run_manifest.get("status") != "complete":
        raise ValueError("source run must be complete")
    run_config = _run_config(run)
    _validate_external_compatibility(run_manifest, run_config, cache)
    checkpoint = _declared_artifact(run, run_manifest, run / "checkpoints" / "best.pt")
    checkpoint_sha = str(run_manifest["artifacts"]["checkpoints/best.pt"]["sha256"])
    snapshot = load_verified_torch(
        checkpoint,
        expected_sha256=checkpoint_sha,
        weights_only=False,
    )
    if not isinstance(snapshot, Mapping):
        raise ValueError("head checkpoint must contain a mapping")
    _checkpoint_identity(
        snapshot,
        run_manifest,
        run_config.model.force.target_mode,
    )
    model = ConfidenceModel(
        force_input_dim=int(run_manifest["force_feature_dim"]),
        energy_input_dim=int(run_manifest["energy_feature_dim"]),
        force_hidden_dims=run_config.model.force.hidden_dims,
        energy_hidden_dims=run_config.model.energy.hidden_dims,
        force_dropout=run_config.model.force.dropout,
        energy_dropout=run_config.model.energy.dropout,
        force_num_bins=run_config.model.force.num_bins,
        energy_num_bins=run_config.model.energy.num_bins,
        cumulant_order=run_config.model.energy.cumulant_order,
        signed_root=run_config.model.energy.signed_root,
        force_target_mode=run_config.model.force.target_mode,
        force_active=run_config.loss.force_coefficient > 0,
        energy_active=run_config.loss.energy_coefficient > 0,
    )
    model.load_state_dict(snapshot["model"])
    device = torch.device(device_name)
    model.to(device).eval()
    loader = DataLoader(
        cache.dataset(),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_cached_structures,
    )
    force_spec, energy_spec = _specs(run_config)
    result = collect_single_target_predictions(
        model=model,
        loader=loader,
        device=device,
        config=run_config,
        force_spec=force_spec,
        energy_spec=energy_spec,
        to_device=_to_device,
        offsets=_offsets,
        include_raw=True,
    )
    order = (
        run_config.model.energy.cumulant_order if result.target == "energy" else None
    )
    payload = {
        "dataset": {
            "name": dataset_name,
            "sha256": cache.dataset_sha256,
        },
        "source_run_id": run_manifest["run_id"],
        "head_checkpoint_sha256": checkpoint_sha,
        "input_cache_id": cache.cache_id,
        "training_cache_id": run_manifest["cache_id"],
        "binning_id": run_manifest["binning_id"],
        "target": result.target,
        "order": order,
    }
    return _publish_external_result(
        run_dir=run,
        dataset_name=dataset_name,
        result=result,
        spec=force_spec if result.target == "force" else energy_spec,
        identity_payload=payload,
    )


def _existing_evaluations(runs: ExternalRuns) -> tuple[Path, ...]:
    outputs: list[Path] = []
    for run in runs.ordered():
        verify_run(run, full=True, allow_plots=True)
        evaluation = run / "evaluation"
        if not (evaluation / "manifest.json").is_file():
            raise ValueError(f"existing evaluation is missing: {run.name}")
        outputs.append(evaluation)
    return tuple(outputs)


def predict_external_datasets(
    config: ExternalPredictionConfig,
    names: Sequence[str] = (),
) -> tuple[Path, ...]:
    """Resolve all selected dataset sources and publish only missing predictions."""

    selected = tuple(names) if names else tuple(config.datasets)
    if not selected or any(name not in config.datasets for name in selected):
        raise ValueError("selected external dataset is not configured")
    runs = discover_external_runs(config.runs_root)
    outputs: list[Path] = []
    for name in selected:
        source = config.datasets[name]
        if isinstance(source, ExistingEvaluationSource):
            outputs.extend(_existing_evaluations(runs))
            continue
        if isinstance(source, ExtXYZSource):
            shared = build_external_dataset_cache(config, name)
            caches = {run: shared for run in runs.ordered()}
        elif isinstance(source, CacheSplitSource):
            caches = {
                run: resolve_declared_cache_split(run, source) for run in runs.ordered()
            }
        else:  # pragma: no cover - strict config exhausts the union
            raise TypeError(f"unsupported dataset source: {type(source)!r}")
        for run in runs.ordered():
            outputs.append(
                predict_dataset_run(
                    run,
                    name,
                    caches[run],
                    batch_size=config.batch_size,
                    device_name=config.device,
                )
            )
    return tuple(outputs)
