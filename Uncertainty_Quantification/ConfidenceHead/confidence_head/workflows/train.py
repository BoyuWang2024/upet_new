"""Train independent force and energy confidence readouts from a raw cache."""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from contextlib import nullcontext
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from ..artifacts import atomic_write_json, load_verified_torch, sha256_file
from ..binning import BinningSpec, fixed_linear_binning, labels_from_thresholds
from ..cache import CachedSplitDataset, collate_cached_structures
from ..config import ConfidenceConfig
from ..errors import energy_per_atom_error, force_component_error
from ..identity import binning_id, config_id, model_loss_id, run_id
from ..losses import LossOutput, confidence_loss
from ..model import ConfidenceModel
from ..trainer import (
    EarlyStoppingState,
    LossAccumulator,
    TrainingIdentity,
    advance_validation_epoch,
    build_plateau_scheduler,
    capture_training_snapshot,
    commit_epoch_checkpoints,
    restore_training_snapshot,
)


RUN_SCHEMA_VERSION = "upet_confidence_run_v1"


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unavailable"


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[4],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return result.stdout.strip()


def _provenance(started_at: str, completed_at: str | None = None) -> dict[str, Any]:
    return {
        "git_commit": _git_commit(),
        "python_version": sys.version.split()[0],
        "dependencies": {
            name: _package_version(name)
            for name in ("torch", "numpy", "pydantic", "metatrain", "upet", "metatomic")
        },
        "started_at": started_at,
        "completed_at": completed_at,
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON {path} must contain a mapping")
    return value


def _validate_cache_identity(
    config: ConfidenceConfig, cache_manifest: Mapping[str, Any]
) -> None:
    payload = cache_manifest.get("identity_payload")
    if not isinstance(payload, Mapping):
        raise ValueError("cache identity_payload must be a mapping")
    checkpoint = payload.get("checkpoint")
    if (
        not isinstance(checkpoint, Mapping)
        or checkpoint.get("sha256") != config.checkpoint.expected_sha256
    ):
        raise ValueError("checkpoint identity disagrees with cache")
    outputs = payload.get("outputs")
    expected_outputs = {
        "force_prediction": config.readouts.force_prediction,
        "energy_prediction": config.readouts.energy_prediction,
        "force_features": config.readouts.force_features,
        "energy_features": config.readouts.energy_features,
    }
    if not isinstance(outputs, Mapping):
        raise ValueError("cache output/readout identity is missing")
    for field, expected in expected_outputs.items():
        if outputs.get(field) != expected:
            raise ValueError(f"cache readout {field} identity mismatch")
    splits = payload.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("cache split identity is missing")
    for name in ("train", "validation", "test"):
        split = splits.get(name)
        if (
            not isinstance(split, Mapping)
            or split.get("sha256") != getattr(config.data, name).expected_sha256
        ):
            raise ValueError(f"cache {name} split identity mismatch")


def _read_metrics(path: Path) -> list[dict[str, int | float]]:
    if not path.is_file():
        raise ValueError("resume metrics artifact is missing")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"unable to read resume metrics {path}: {error}") from error
    records: list[dict[str, int | float]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid resume metrics JSON at line {line_number}: {error}"
            ) from error
        if not isinstance(record, dict):
            raise ValueError(
                f"resume metrics line {line_number} must contain a JSON mapping"
            )
        records.append(record)
    return records


def _validate_run_name(run_name: str) -> None:
    if (
        not isinstance(run_name, str)
        or not run_name
        or run_name in {".", ".."}
        or Path(run_name).name != run_name
    ):
        raise ValueError(f"unsafe run name: {run_name!r}")


def _safe_run_dir(root: Path, run_name: str) -> Path:
    _validate_run_name(run_name)
    run_dir = Path(root) / "runs" / run_name
    if run_dir.exists():
        raise ValueError(f"run directory already exists: {run_dir}")
    return run_dir


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _bin_payload(force: BinningSpec, energy: BinningSpec) -> dict[str, Any]:
    def branch(spec: BinningSpec) -> dict[str, Any]:
        return {
            "algorithm": spec.algorithm,
            "num_bins": spec.num_bins,
            "max_error": spec.max_error,
            "thresholds": spec.thresholds.tolist(),
            "representatives": spec.representatives.tolist(),
        }

    return {"force": branch(force), "energy": branch(energy)}


def _identities(
    config: ConfidenceConfig,
    cache_manifest: Mapping[str, Any],
    bins: Mapping[str, Any],
) -> tuple[TrainingIdentity, str, dict[str, Any]]:
    resolved = config.model_dump(mode="json")
    config_identity = config_id(resolved)
    bin_identity = binning_id(dict(bins))
    model_loss_payload = {
        "model": resolved["model"],
        "loss": resolved["loss"],
        "force_num_bins": resolved["binning"]["force_num_bins"],
        "energy_num_bins": resolved["binning"]["energy_num_bins"],
    }
    model_identity = model_loss_id(model_loss_payload)
    training = TrainingIdentity(
        config_id=config_identity,
        cache_id=str(cache_manifest["cache_id"]),
        binning_id=bin_identity,
        model_loss_id=model_identity,
    )
    run_identity = run_id(
        {
            "config_id": training.config_id,
            "cache_id": training.cache_id,
            "binning_id": training.binning_id,
            "model_loss_id": training.model_loss_id,
        }
    )
    return training, run_identity, resolved


def _loader(
    dataset: CachedSplitDataset,
    config: ConfidenceConfig,
    *,
    shuffle: bool,
    generator: torch.Generator,
) -> DataLoader[dict[str, Any]]:
    return DataLoader(
        dataset,
        batch_size=config.cache.batch_size,
        shuffle=shuffle,
        num_workers=config.cache.num_workers,
        collate_fn=collate_cached_structures,
        generator=generator,
    )


def _to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _batch_loss(
    model: ConfidenceModel,
    batch: Mapping[str, Any],
    force_spec: BinningSpec,
    energy_spec: BinningSpec,
    config: ConfidenceConfig,
) -> LossOutput:
    force_observed = force_component_error(
        batch["force_prediction"], batch["force_reference"]
    )
    energy_observed = energy_per_atom_error(
        batch["energy_prediction"],
        batch["energy_reference"],
        batch["num_atoms"],
    )
    force_labels = labels_from_thresholds(force_observed, force_spec.thresholds)
    energy_labels = labels_from_thresholds(energy_observed, energy_spec.thresholds)
    output = model(
        batch["force_features"],
        batch["energy_features"],
        batch["atom_offsets"],
    )
    return confidence_loss(
        output.force_logits,
        force_labels,
        output.energy_logits,
        energy_labels,
        force_weight=config.loss.force_coefficient,
        energy_weight=config.loss.energy_coefficient,
    )


def _accumulate(accumulator: LossAccumulator, loss: LossOutput) -> None:
    accumulator.update(
        force_loss_sum=float(loss.force.detach()) * loss.force_count,
        force_count=loss.force_count,
        energy_loss_sum=float(loss.energy.detach()) * loss.energy_count,
        energy_count=loss.energy_count,
    )


def _epoch(
    *,
    model: ConfidenceModel,
    loader: DataLoader[dict[str, Any]],
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    force_spec: BinningSpec,
    energy_spec: BinningSpec,
    config: ConfidenceConfig,
) -> tuple[float, float, float, int]:
    training = optimizer is not None
    model.train(training)
    accumulator = LossAccumulator()
    steps = 0
    amp_enabled = config.run.amp
    grad_context = nullcontext() if training else torch.inference_mode()
    with grad_context:
        for raw_batch in loader:
            batch = _to_device(raw_batch, device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            autocast = torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            )
            with autocast:
                loss = _batch_loss(model, batch, force_spec, energy_spec, config)
            if optimizer is not None:
                loss.total.backward()
                optimizer.step()
            _accumulate(accumulator, loss)
            steps += 1
    force, energy, total = accumulator.result(
        config.loss.force_coefficient,
        config.loss.energy_coefficient,
    )
    return force, energy, total, steps


def _artifact_entry(run_dir: Path, relative: str) -> dict[str, str]:
    path = run_dir / relative
    return {"path": relative, "sha256": sha256_file(path)}


def _resume_artifact(
    run_dir: Path,
    manifest: Mapping[str, Any],
    relative: str,
) -> tuple[Path, str]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("resume run has no declared artifacts")
    entry = artifacts.get(relative)
    if not isinstance(entry, Mapping) or entry.get("path") != relative:
        raise ValueError(f"resume artifact is not declared: {relative}")
    path = (run_dir / relative).resolve()
    if not path.is_relative_to(run_dir.resolve()) or not path.is_file():
        raise ValueError(f"resume artifact is missing or unsafe: {relative}")
    expected = entry.get("sha256")
    if not isinstance(expected, str) or sha256_file(path) != expected:
        raise ValueError(f"resume metrics sha256 mismatch: {relative}")
    return path, expected


def _validate_resume_metrics(
    records: list[dict[str, int | float]],
    *,
    restored_epoch: int,
    global_step: int,
    learning_rate: float,
    ema: float | None,
) -> None:
    if len(records) != restored_epoch + 1:
        raise ValueError("resume metrics count disagrees with checkpoint epoch")
    for expected_epoch, record in enumerate(records):
        if record.get("epoch") != expected_epoch:
            raise ValueError("resume metrics epoch sequence is not contiguous")
    last = records[-1]
    boundaries = (
        ("epoch", restored_epoch, "epoch"),
        ("global_step", global_step, "global_step"),
        ("learning_rate", learning_rate, "learning_rate"),
        ("val/total_loss_ema", ema, "EMA"),
    )
    for metric_field, expected, label in boundaries:
        if last.get(metric_field) != expected:
            raise ValueError(f"resume metrics {label} disagrees with checkpoint")


def train_run(
    config: ConfidenceConfig,
    *,
    cache_manifest_path: Path,
    run_name: str,
    stop_after_epoch: int | None = None,
    resume_from: Path | None = None,
) -> Path:
    """Train confidence heads and atomically publish a complete run manifest."""
    cache_path = Path(cache_manifest_path).resolve()
    cache_manifest = _load_json(cache_path)
    if cache_manifest.get("status") != "complete" or not isinstance(
        cache_manifest.get("cache_id"), str
    ):
        raise ValueError("cache manifest must be complete and identified")
    _validate_cache_identity(config, cache_manifest)
    cache_identity = str(cache_manifest["cache_id"])
    split_metadata = cache_manifest.get("splits")
    if not isinstance(split_metadata, dict):
        raise ValueError("cache manifest splits must be a mapping")
    for split in ("train", "validation", "test"):
        if split not in split_metadata:
            raise ValueError(f"cache manifest is missing {split!r}")

    force_spec = fixed_linear_binning(
        config.binning.force_num_bins, config.binning.force_max_error
    )
    energy_spec = fixed_linear_binning(
        config.binning.energy_num_bins, config.binning.energy_max_error
    )
    bins = _bin_payload(force_spec, energy_spec)
    identity, run_identity, resolved = _identities(config, cache_manifest, bins)
    _validate_run_name(run_name)
    run_dir = Path(config.run.output_root) / "runs" / run_name
    if resume_from is None:
        _safe_run_dir(config.run.output_root, run_name)
    run_dir.mkdir(parents=True, exist_ok=resume_from is not None)
    manifest_path = run_dir / "manifest.json"
    started_at = datetime.now(UTC).isoformat()
    base_manifest: dict[str, Any] = {
        "schema_version": RUN_SCHEMA_VERSION,
        "status": "incomplete",
        "identity": run_identity,
        "run_id": run_identity,
        "config_id": identity.config_id,
        "cache_id": identity.cache_id,
        "binning_id": identity.binning_id,
        "model_loss_id": identity.model_loss_id,
        "profile": config.profile,
        "cache_manifest": str(cache_path),
        "readouts": resolved["readouts"],
        "execution": resolved["run"],
        "source_identity": {
            "checkpoint": resolved["checkpoint"],
            "splits": cache_manifest["identity_payload"]["splits"],
        },
        "provenance": _provenance(started_at),
        "artifacts": {},
    }
    previous_manifest: dict[str, Any] | None = None
    if resume_from is not None:
        previous_manifest = _load_json(manifest_path)
        for field, expected in (
            ("run_id", run_identity),
            ("config_id", identity.config_id),
            ("cache_id", identity.cache_id),
            ("binning_id", identity.binning_id),
            ("model_loss_id", identity.model_loss_id),
        ):
            if previous_manifest.get(field) != expected:
                raise ValueError(f"resume run {field} identity mismatch")
        previous_provenance = previous_manifest.get("provenance")
        previous_start = (
            previous_provenance.get("started_at")
            if isinstance(previous_provenance, Mapping)
            else None
        )
        if isinstance(previous_start, str):
            started_at = previous_start
            base_manifest["provenance"] = _provenance(started_at)
    if resume_from is None:
        atomic_write_json(manifest_path, base_manifest)
        _atomic_write_text(
            run_dir / "resolved_config.yaml",
            yaml.safe_dump(resolved, sort_keys=True),
        )
        atomic_write_json(run_dir / "binning.json", bins)

    train_data = CachedSplitDataset(cache_path, "train", cache_identity)
    validation_data = CachedSplitDataset(cache_path, "validation", cache_identity)
    train_meta = split_metadata["train"]
    if not isinstance(train_meta, dict):
        raise ValueError("train cache metadata must be a mapping")
    force_dim = int(train_meta["force_feature_dim"])
    energy_dim = int(train_meta["energy_feature_dim"])

    random.seed(config.run.seed)
    np.random.seed(config.run.seed)
    torch.manual_seed(config.run.seed)
    device = torch.device(config.run.device)
    if config.run.amp and device.type != "cuda":
        raise ValueError("amp=true requires a CUDA device")
    model = ConfidenceModel(
        force_input_dim=force_dim,
        energy_input_dim=energy_dim,
        hidden_dims=config.model.hidden_dims,
        force_num_bins=config.binning.force_num_bins,
        num_bins=config.binning.force_num_bins,
        energy_num_bins=config.binning.energy_num_bins,
        cumulant_order=config.model.cumulant_order,
        signed_root=config.model.signed_root,
        dropout=config.model.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay,
    )
    scheduler = build_plateau_scheduler(optimizer)
    sampler = torch.Generator(device="cpu")
    sampler.manual_seed(config.run.seed)
    train_loader = _loader(train_data, config, shuffle=True, generator=sampler)
    validation_generator = torch.Generator(device="cpu")
    validation_generator.manual_seed(config.run.seed)
    validation_loader = _loader(
        validation_data,
        config,
        shuffle=False,
        generator=validation_generator,
    )

    control = EarlyStoppingState()
    global_step = 0
    best_step: int | None = None
    metric_records: list[dict[str, int | float]] = []
    start_epoch = 0
    if resume_from is not None:
        assert previous_manifest is not None
        checkpoint = Path(resume_from).resolve()
        if not checkpoint.is_relative_to(run_dir.resolve()):
            raise ValueError("resume checkpoint escapes run directory")
        declared = previous_manifest.get("artifacts")
        if not isinstance(declared, Mapping):
            raise ValueError("resume run has no declared artifacts")
        matching = [
            entry
            for entry in declared.values()
            if isinstance(entry, Mapping)
            and (run_dir / str(entry.get("path"))).resolve() == checkpoint
        ]
        if not matching:
            raise ValueError("resume checkpoint is not declared by run manifest")
        if not checkpoint.is_file() or sha256_file(checkpoint) != matching[0].get(
            "sha256"
        ):
            raise ValueError("resume checkpoint sha256 mismatch")
        snapshot = load_verified_torch(
            checkpoint,
            expected_sha256=str(matching[0]["sha256"]),
            weights_only=False,
        )
        if not isinstance(snapshot, Mapping):
            raise ValueError("resume checkpoint must contain a mapping")
        restored = restore_training_snapshot(
            snapshot=snapshot,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_identity=identity,
            sampler_generator=sampler,
        )
        start_epoch = restored.next_epoch
        global_step = restored.global_step
        control = restored.control_state
        best_step = restored.best_step
        metrics_path, _ = _resume_artifact(
            run_dir, previous_manifest, "logs/metrics.jsonl"
        )
        metric_records = _read_metrics(metrics_path)
        _validate_resume_metrics(
            metric_records,
            restored_epoch=restored.epoch,
            global_step=restored.global_step,
            learning_rate=restored.learning_rate,
            ema=restored.control_state.ema,
        )
    externally_stopped = False
    for epoch in range(start_epoch, config.trainer.max_epochs):
        train_force, train_energy, train_total, steps = _epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            force_spec=force_spec,
            energy_spec=energy_spec,
            config=config,
        )
        global_step += steps
        val_force, val_energy, val_total, _ = _epoch(
            model=model,
            loader=validation_loader,
            optimizer=None,
            device=device,
            force_spec=force_spec,
            energy_spec=energy_spec,
            config=config,
        )
        update = advance_validation_epoch(
            state=control,
            raw_total_loss=val_total,
            epoch=epoch,
            scheduler=scheduler,
            beta=config.trainer.ema_beta,
            min_delta=config.trainer.min_delta,
            patience=config.trainer.early_stopping_patience,
            min_epochs=config.trainer.min_epochs,
        )
        control = update.state
        if update.improved:
            best_step = global_step
        snapshot = capture_training_snapshot(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            control_state=control,
            best_step=best_step,
            identity=identity,
            sampler_generator=sampler,
        )
        external_stop = commit_epoch_checkpoints(
            checkpoint_dir=run_dir / "checkpoints",
            snapshot=snapshot,
            save_best=update.should_save_best,
            profile=config.profile,
            stop_after_epoch=stop_after_epoch,
            max_epochs=config.trainer.max_epochs,
        )
        externally_stopped = external_stop
        assert control.ema is not None
        record: dict[str, int | float] = {
            "epoch": epoch,
            "global_step": global_step,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train/force_loss": train_force,
            "train/energy_loss": train_energy,
            "train/total_loss": train_total,
            "val/force_loss": val_force,
            "val/energy_loss": val_energy,
            "val/total_loss": val_total,
            "val/total_loss_ema": control.ema,
        }
        metric_records.append(record)
        _atomic_write_text(
            run_dir / "logs" / "metrics.jsonl",
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in metric_records),
        )
        if update.should_stop or external_stop:
            break

    training_artifacts = {
        relative: _artifact_entry(run_dir, relative)
        for relative in (
            "resolved_config.yaml",
            "binning.json",
            "checkpoints/best.pt",
            "checkpoints/last.pt",
            "logs/metrics.jsonl",
        )
    }
    complete = {
        **base_manifest,
        "status": "complete",
        "artifacts": training_artifacts,
        "best_epoch": control.best_epoch,
        "best_metric": control.best,
        "stop_reason": (
            "external_stop_after_epoch"
            if externally_stopped
            else control.stop_reason
            if control.stop_reason
            else "max_epochs"
        ),
        "epochs_completed": len(metric_records),
        "force_feature_dim": force_dim,
        "energy_feature_dim": energy_dim,
        "max_epochs": config.trainer.max_epochs,
    }
    complete["provenance"] = _provenance(
        started_at,
        datetime.now(UTC).isoformat(),
    )
    atomic_write_json(manifest_path, complete)
    return run_dir
