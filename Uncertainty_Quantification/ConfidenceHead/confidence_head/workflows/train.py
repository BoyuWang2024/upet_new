"""Train independent force and energy confidence readouts from a raw cache."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, nullcontext
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from ..artifacts import atomic_write_json, load_verified_torch
from ..artifacts import sha256_file as sha256_file
from ..binning import BinningSpec, fixed_linear_binning, labels_from_thresholds
from ..cache import CachedSplitDataset, collate_cached_structures
from ..config import ConfidenceConfig, ForceTargetMode
from ..errors import (
    energy_per_atom_error,
    force_component_error,
    force_error_definition,
)
from ..identity import binning_id, config_id, model_loss_id, run_id
from ..losses import LossOutput, confidence_loss
from ..model import ConfidenceModel
from ..tracking import Tracker, TrackerFactory, WandbTracker
from ..trainer import (
    CHECKPOINT_SCHEMA_VERSION,
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


def _read_metrics(data: bytes) -> list[dict[str, int | float]]:
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"invalid resume metrics UTF-8: {error}") from error
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


def _safe_run_dir(root: Path, run_name: str, *, resume: bool) -> Path:
    _validate_run_name(run_name)
    output_root = Path(root).resolve()
    runs_root = Path(root) / "runs"
    if runs_root.is_symlink():
        raise ValueError("runs directory must not be a symlink")
    resolved_runs_root = runs_root.resolve()
    if not resolved_runs_root.is_relative_to(output_root):
        raise ValueError("runs directory escapes output root")
    run_dir = runs_root / run_name
    if run_dir.is_symlink():
        raise ValueError("run directory must not be a symlink")
    resolved_run_dir = run_dir.resolve()
    if not resolved_run_dir.is_relative_to(resolved_runs_root):
        raise ValueError("run directory escapes runs root")
    if not resume and run_dir.exists():
        raise ValueError(f"run directory already exists: {run_dir}")
    if resume and not run_dir.is_dir():
        raise ValueError(f"resume run directory is missing: {run_dir}")
    return run_dir


@contextmanager
def _exclusive_run_lock(root: Path, run_name: str) -> Iterator[None]:
    """Serialize cooperative train_run writers for one run name."""
    _validate_run_name(run_name)
    output_root = Path(root).resolve()
    runs_root = Path(root) / "runs"
    if runs_root.is_symlink():
        raise ValueError("runs directory must not be a symlink")
    runs_root.mkdir(parents=True, exist_ok=True)
    if runs_root.is_symlink() or not runs_root.resolve().is_relative_to(output_root):
        raise ValueError("runs directory escapes output root")
    lock_root = runs_root / ".locks"
    if lock_root.is_symlink():
        raise ValueError("run lock directory must not be a symlink")
    lock_root.mkdir(exist_ok=True)
    if lock_root.is_symlink():
        raise ValueError("run lock directory must not be a symlink")
    lock_name = hashlib.sha256(run_name.encode("utf-8")).hexdigest() + ".lock"
    lock_path = lock_root / lock_name
    if lock_path.is_symlink():
        raise ValueError("run lock file must not be a symlink")
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    handle = os.fdopen(descriptor, "r+b", buffering=0)
    locked = False
    try:
        try:
            if os.name == "nt":
                import msvcrt

                if os.fstat(handle.fileno()).st_size == 0:
                    handle.write(b"\0")
                handle.seek(0)
                msvcrt.locking(  # type: ignore[attr-defined]
                    handle.fileno(),
                    msvcrt.LK_NBLCK,  # type: ignore[attr-defined]
                    1,
                )
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError as error:
            raise ValueError(
                f"an active writer holds the run lock for {run_name!r}"
            ) from error
        yield
    finally:
        if locked:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(  # type: ignore[attr-defined]
                    handle.fileno(),
                    msvcrt.LK_UNLCK,  # type: ignore[attr-defined]
                    1,
                )
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _run_directory_identity(run_dir: Path, output_root: Path) -> tuple[int, int]:
    if run_dir.is_symlink():
        raise ValueError("run directory must not be a symlink")
    resolved = run_dir.resolve()
    expected_root = (Path(output_root).resolve() / "runs").resolve()
    if not resolved.is_relative_to(expected_root):
        raise ValueError("run directory escapes output root")
    stat = run_dir.stat(follow_symlinks=False)
    return stat.st_dev, stat.st_ino


def _assert_run_directory_identity(
    run_dir: Path,
    output_root: Path,
    expected: tuple[int, int],
) -> None:
    if _run_directory_identity(run_dir, output_root) != expected:
        raise ValueError("run directory identity changed during publication")


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


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


_TRAINING_ARTIFACTS = (
    "resolved_config.yaml",
    "binning.json",
    "checkpoints/best.pt",
    "checkpoints/last.pt",
    "logs/metrics.jsonl",
)


def _bin_payload(
    force: BinningSpec,
    energy: BinningSpec,
    force_target_mode: ForceTargetMode,
) -> dict[str, Any]:
    def branch(spec: BinningSpec) -> dict[str, Any]:
        return {
            "algorithm": spec.algorithm,
            "num_bins": spec.num_bins,
            "max_error": spec.max_error,
            "thresholds": spec.thresholds.tolist(),
            "representatives": spec.representatives.tolist(),
        }

    force_payload = branch(force)
    if force_target_mode == "atom_mean":
        force_payload.update(
            {
                "target_mode": force_target_mode,
                "error_definition": force_error_definition(force_target_mode),
            }
        )
    return {"force": force_payload, "energy": branch(energy)}


def _resolved_experiment_config(config: ConfidenceConfig) -> dict[str, Any]:
    """Return semantic experiment config without resume control state."""
    resolved = config.model_dump(mode="json")
    resolved["trainer"]["resume_from"] = None
    return resolved


def _identities(
    config: ConfidenceConfig,
    cache_manifest: Mapping[str, Any],
    bins: Mapping[str, Any],
) -> tuple[TrainingIdentity, str, dict[str, Any]]:
    resolved = _resolved_experiment_config(config)
    config_identity = config_id(resolved)
    bin_identity = binning_id(dict(bins))
    model_loss_payload = {
        "model": resolved["model"],
        "loss": resolved["loss"],
        "force_num_bins": resolved["model"]["force"]["num_bins"],
        "energy_num_bins": resolved["model"]["energy"]["num_bins"],
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
        batch_size=config.trainer.batch_size,
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


def _resume_artifact_bytes(
    run_dir: Path,
    manifest: Mapping[str, Any],
    relative: str,
) -> tuple[Path, str, bytes]:
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
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ValueError(
            f"unable to read resume artifact {relative}: {error}"
        ) from error
    actual = hashlib.sha256(data).hexdigest()
    if not isinstance(expected, str) or actual != expected:
        raise ValueError(f"resume artifact {relative} sha256 mismatch")
    return path, expected, data


def _validate_checkpoint_identity(
    path: Path,
    entry: Mapping[str, str],
    identity: TrainingIdentity,
) -> Mapping[str, Any]:
    try:
        snapshot = load_verified_torch(
            path,
            expected_sha256=entry["sha256"],
            weights_only=False,
        )
    except Exception as error:
        raise ValueError(f"invalid checkpoint artifact {path}: {error}") from error
    if not isinstance(snapshot, Mapping):
        raise ValueError(f"checkpoint artifact must be a mapping: {path}")
    if snapshot.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"checkpoint artifact schema mismatch: {path}")
    for field in ("config_id", "cache_id", "binning_id", "model_loss_id"):
        if snapshot.get(field) != getattr(identity, field):
            raise ValueError(f"checkpoint artifact {field} mismatch: {path}")
    return snapshot


def _validate_publish_artifacts(
    *,
    run_dir: Path,
    resolved: Mapping[str, Any],
    bins: Mapping[str, Any],
    identity: TrainingIdentity,
    resume_artifacts: Mapping[str, tuple[Path, str, bytes]],
    best_rewritten: bool,
) -> dict[str, dict[str, str]]:
    relatives = (
        "resolved_config.yaml",
        "binning.json",
        "checkpoints/best.pt",
        "checkpoints/last.pt",
        "logs/metrics.jsonl",
    )
    payloads: dict[str, bytes] = {}
    artifacts: dict[str, dict[str, str]] = {}
    for relative in relatives:
        path = run_dir / relative
        try:
            data = path.read_bytes()
        except OSError as error:
            raise ValueError(
                f"unable to read artifact before publish {relative}: {error}"
            ) from error
        payloads[relative] = data
        artifacts[relative] = {
            "path": relative,
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    try:
        published_resolved = yaml.safe_load(
            payloads["resolved_config.yaml"].decode("utf-8")
        )
        published_bins = json.loads(payloads["binning.json"].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, yaml.YAMLError) as error:
        raise ValueError(f"invalid static artifact before publish: {error}") from error
    if published_resolved != resolved or published_bins != bins:
        raise ValueError("static artifact changed before publish")
    if resume_artifacts:
        for relative in ("resolved_config.yaml", "binning.json"):
            if artifacts[relative]["sha256"] != resume_artifacts[relative][1]:
                raise ValueError(f"resume artifact changed before publish: {relative}")
        if (
            not best_rewritten
            and artifacts["checkpoints/best.pt"]["sha256"]
            != resume_artifacts["checkpoints/best.pt"][1]
        ):
            raise ValueError("best checkpoint artifact changed before publish")
    best = _validate_checkpoint_identity(
        run_dir / "checkpoints" / "best.pt",
        artifacts["checkpoints/best.pt"],
        identity,
    )
    last = _validate_checkpoint_identity(
        run_dir / "checkpoints" / "last.pt",
        artifacts["checkpoints/last.pt"],
        identity,
    )
    del best
    metrics = _read_metrics(payloads["logs/metrics.jsonl"])
    try:
        _validate_resume_metrics(
            metrics,
            restored_epoch=int(last["epoch"]),
            global_step=int(last["global_step"]),
            learning_rate=float(last["learning_rate"]),
            ema=None if last["ema"] is None else float(last["ema"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"metrics and last checkpoint disagree before publish: {error}"
        ) from error
    return artifacts


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


def _train_run_locked(
    config: ConfidenceConfig,
    *,
    cache_manifest_path: Path,
    run_name: str,
    stop_after_epoch: int | None = None,
    resume_from: Path | None = None,
    begin_resume_writes: Callable[[], None] | None = None,
    tracker_factory: TrackerFactory,
    on_tracker_started: Callable[[Tracker], None],
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
        config.model.force.num_bins, config.binning.force_max_error
    )
    energy_spec = fixed_linear_binning(
        config.model.energy.num_bins, config.binning.energy_max_error
    )
    bins = _bin_payload(force_spec, energy_spec, config.model.force.target_mode)
    identity, run_identity, resolved = _identities(config, cache_manifest, bins)
    run_dir = _safe_run_dir(
        config.run.output_root, run_name, resume=resume_from is not None
    )
    if resume_from is None:
        run_dir.mkdir(parents=True)
    run_directory_identity = _run_directory_identity(run_dir, config.run.output_root)
    manifest_path = run_dir / "manifest.json"
    started_at = datetime.now(UTC).isoformat()
    resume_started_at = started_at
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
    previous_history: list[dict[str, Any]] = []
    resume_artifacts: dict[str, tuple[Path, str, bytes]] = {}
    stale_evaluation_artifacts: tuple[str, ...] = ()
    if resume_from is not None:
        previous_manifest = _load_json(manifest_path)
        if (
            previous_manifest.get("schema_version") != RUN_SCHEMA_VERSION
            or previous_manifest.get("status") != "complete"
            or previous_manifest.get("identity") != run_identity
        ):
            raise ValueError("resume run schema/status/identity mismatch")
        for field, expected in (
            ("run_id", run_identity),
            ("config_id", identity.config_id),
            ("cache_id", identity.cache_id),
            ("binning_id", identity.binning_id),
            ("model_loss_id", identity.model_loss_id),
        ):
            if previous_manifest.get(field) != expected:
                raise ValueError(f"resume run {field} identity mismatch")
        declared = previous_manifest.get("artifacts")
        if not isinstance(declared, Mapping):
            raise ValueError("resume run has no declared artifacts")
        for relative in declared:
            if not isinstance(relative, str):
                raise ValueError("resume artifact names must be strings")
            if relative not in _TRAINING_ARTIFACTS and not relative.startswith(
                "evaluation/"
            ):
                raise ValueError(f"unsupported resume artifact: {relative}")
            resume_artifacts[relative] = _resume_artifact_bytes(
                run_dir, previous_manifest, relative
            )
        for relative in _TRAINING_ARTIFACTS:
            if relative not in resume_artifacts:
                raise ValueError(f"resume artifact is not declared: {relative}")
        stale_evaluation_artifacts = tuple(
            sorted(
                relative
                for relative in resume_artifacts
                if relative.startswith("evaluation/")
            )
        )
        if stale_evaluation_artifacts:
            evaluation = previous_manifest.get("evaluation")
            if (
                not isinstance(evaluation, Mapping)
                or evaluation.get("status") != "complete"
                or evaluation.get("manifest") not in stale_evaluation_artifacts
            ):
                raise ValueError("resume evaluation declaration is invalid")
            evaluation_manifest = json.loads(
                resume_artifacts[str(evaluation["manifest"])][2].decode("utf-8")
            )
            if not isinstance(evaluation_manifest, Mapping):
                raise ValueError("resume evaluation manifest must be a mapping")
        elif "evaluation" in previous_manifest:
            raise ValueError("resume evaluation declaration has no artifacts")
        try:
            previous_resolved = yaml.safe_load(
                resume_artifacts["resolved_config.yaml"][2].decode("utf-8")
            )
        except (UnicodeDecodeError, yaml.YAMLError) as error:
            raise ValueError(
                f"invalid resume resolved_config artifact: {error}"
            ) from error
        if previous_resolved != resolved:
            raise ValueError("resume resolved_config artifact disagrees with config")
        try:
            previous_bins = json.loads(
                resume_artifacts["binning.json"][2].decode("utf-8")
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid resume binning artifact: {error}") from error
        if previous_bins != bins:
            raise ValueError("resume binning artifact disagrees with config")
        previous_provenance = previous_manifest.get("provenance")
        if not isinstance(previous_provenance, Mapping):
            raise ValueError("resume provenance must be a mapping")
        previous_start = previous_provenance.get("started_at")
        if isinstance(previous_start, str):
            started_at = previous_start
            base_manifest["provenance"] = _provenance(started_at)
        raw_history = previous_provenance.get("history", [])
        if not isinstance(raw_history, list):
            raise ValueError("resume provenance history must be a list")
        previous_history = list(raw_history)
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
        force_hidden_dims=config.model.force.hidden_dims,
        energy_hidden_dims=config.model.energy.hidden_dims,
        force_dropout=config.model.force.dropout,
        energy_dropout=config.model.energy.dropout,
        force_num_bins=config.model.force.num_bins,
        energy_num_bins=config.model.energy.num_bins,
        cumulant_order=config.model.energy.cumulant_order,
        signed_root=config.model.energy.signed_root,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
    )
    scheduler = build_plateau_scheduler(
        optimizer,
        factor=config.scheduler.factor,
        patience=config.scheduler.patience,
        threshold=config.scheduler.threshold,
        threshold_mode=config.scheduler.threshold_mode,
        cooldown=config.scheduler.cooldown,
        min_lr=config.scheduler.min_lr,
    )
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
    best_rewritten = False
    if resume_from is not None:
        assert previous_manifest is not None
        checkpoint = Path(resume_from).resolve()
        if not checkpoint.is_relative_to(run_dir.resolve()):
            raise ValueError("resume checkpoint escapes run directory")
        matching = [
            artifact
            for relative, artifact in resume_artifacts.items()
            if relative.startswith("checkpoints/") and artifact[0] == checkpoint
        ]
        if not matching:
            raise ValueError("resume checkpoint is not declared by run manifest")
        snapshot = load_verified_torch(
            checkpoint,
            expected_sha256=matching[0][1],
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
        metric_records = _read_metrics(resume_artifacts["logs/metrics.jsonl"][2])
        _validate_resume_metrics(
            metric_records,
            restored_epoch=restored.epoch,
            global_step=restored.global_step,
            learning_rate=restored.learning_rate,
            ema=restored.control_state.ema,
        )

    resume_id: str | None = None
    if previous_manifest is not None:
        if "tracking" not in previous_manifest:
            previous_tracking: Any = {
                "wandb_enabled": config.logging.wandb,
                "wandb_mode": config.logging.wandb_mode,
                "wandb_project": config.logging.wandb_project,
                "wandb_run_id": None,
            }
        else:
            previous_tracking = previous_manifest["tracking"]
        if not isinstance(previous_tracking, Mapping):
            raise ValueError("resume tracking declaration must be a mapping")
        tracking_expectations: tuple[tuple[str, object], ...] = (
            ("wandb_enabled", config.logging.wandb),
            ("wandb_mode", config.logging.wandb_mode),
            ("wandb_project", config.logging.wandb_project),
        )
        for field, tracking_expected in tracking_expectations:
            if previous_tracking.get(field) != tracking_expected:
                raise ValueError(f"resume tracking {field} mismatch")
        raw_resume_id = previous_tracking.get("wandb_run_id")
        if raw_resume_id is not None and not isinstance(raw_resume_id, str):
            raise ValueError("resume W&B run ID must be a string or null")
        resume_id = raw_resume_id

    tracker = tracker_factory(
        config.logging,
        run_name=run_name,
        resolved_config=resolved,
        resume_id=resume_id,
    )
    on_tracker_started(tracker)
    base_manifest["tracking"] = {
        "wandb_enabled": config.logging.wandb,
        "wandb_mode": config.logging.wandb_mode,
        "wandb_project": config.logging.wandb_project,
        "wandb_run_id": tracker.run_id,
    }
    if resume_from is None:
        _assert_run_directory_identity(
            run_dir, config.run.output_root, run_directory_identity
        )
        atomic_write_json(manifest_path, base_manifest)
        _atomic_write_text(
            run_dir / "resolved_config.yaml",
            yaml.safe_dump(resolved, sort_keys=True),
        )
        atomic_write_json(run_dir / "binning.json", bins)
    else:
        if begin_resume_writes is None:
            raise RuntimeError("resume transaction callback is missing")
        begin_resume_writes()
        if stale_evaluation_artifacts:
            for relative in stale_evaluation_artifacts:
                resume_artifacts[relative][0].unlink()

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
        if begin_resume_writes is not None:
            begin_resume_writes()
        _assert_run_directory_identity(
            run_dir, config.run.output_root, run_directory_identity
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
        if update.should_save_best:
            best_rewritten = True
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
        _assert_run_directory_identity(
            run_dir, config.run.output_root, run_directory_identity
        )
        _atomic_write_text(
            run_dir / "logs" / "metrics.jsonl",
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in metric_records),
        )
        tracker.log(record)
        if update.should_stop or external_stop:
            break

    _assert_run_directory_identity(
        run_dir, config.run.output_root, run_directory_identity
    )
    training_artifacts = _validate_publish_artifacts(
        run_dir=run_dir,
        resolved=resolved,
        bins=bins,
        identity=identity,
        resume_artifacts=resume_artifacts,
        best_rewritten=best_rewritten,
    )
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
    completed_at = datetime.now(UTC).isoformat()
    complete["provenance"] = _provenance(
        started_at,
        completed_at,
    )
    if resume_from is not None:
        complete["provenance"]["history"] = [
            *previous_history,
            {
                "started_at": resume_started_at,
                "completed_at": completed_at,
            },
        ]
    if begin_resume_writes is not None:
        begin_resume_writes()
    _assert_run_directory_identity(
        run_dir, config.run.output_root, run_directory_identity
    )
    atomic_write_json(manifest_path, complete)
    _assert_run_directory_identity(
        run_dir, config.run.output_root, run_directory_identity
    )
    return run_dir


def _snapshot_resume_artifacts(run_dir: Path) -> dict[str, bytes | None]:
    manifest_path = run_dir / "manifest.json"
    manifest_data = manifest_path.read_bytes()
    manifest = json.loads(manifest_data.decode("utf-8"))
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("resume run has no declared artifacts")
    relatives = {"manifest.json"}
    for relative in artifacts:
        if not isinstance(relative, str):
            raise ValueError("resume artifact names must be strings")
        path = (run_dir / relative).resolve()
        if not path.is_relative_to(run_dir.resolve()):
            raise ValueError(f"resume artifact escapes run directory: {relative}")
        relatives.add(relative)
    snapshot: dict[str, bytes | None] = {}
    for relative in sorted(relatives):
        path = run_dir / relative
        snapshot[relative] = (
            manifest_data if relative == "manifest.json" else path.read_bytes()
        )
    return snapshot


def _restore_resume_artifacts(
    run_dir: Path,
    snapshot: Mapping[str, bytes | None],
) -> None:
    for relative, data in snapshot.items():
        path = run_dir / relative
        if data is None:
            path.unlink(missing_ok=True)
        else:
            _atomic_write_bytes(path, data)


def train_run(
    config: ConfidenceConfig,
    *,
    cache_manifest_path: Path,
    run_name: str,
    stop_after_epoch: int | None = None,
    resume_from: Path | None = None,
    tracker_factory: TrackerFactory = WandbTracker.start,
) -> Path:
    """Run one cooperative, per-run locked training transaction.

    The lock serializes all writers using this API. Arbitrary same-user
    processes that ignore the lock protocol are outside the threat model.
    """
    with _exclusive_run_lock(config.run.output_root, run_name):
        rollback: dict[str, bytes | None] | None = None
        rollback_identity: tuple[int, int] | None = None
        run_dir: Path | None = None
        tracker: Tracker | None = None
        finish_summary: dict[str, Any] = {"stop_reason": "exception"}
        finish_status = "failed"
        if resume_from is not None:
            run_dir = _safe_run_dir(
                config.run.output_root,
                run_name,
                resume=True,
            )
            rollback_identity = _run_directory_identity(
                run_dir,
                config.run.output_root,
            )

        def begin_resume_writes() -> None:
            nonlocal rollback
            if rollback is None and run_dir is not None:
                rollback = _snapshot_resume_artifacts(run_dir)

        def on_tracker_started(started_tracker: Tracker) -> None:
            nonlocal tracker
            tracker = started_tracker

        try:
            try:
                result = _train_run_locked(
                    config,
                    cache_manifest_path=cache_manifest_path,
                    run_name=run_name,
                    stop_after_epoch=stop_after_epoch,
                    resume_from=resume_from,
                    begin_resume_writes=begin_resume_writes
                    if run_dir is not None
                    else None,
                    tracker_factory=tracker_factory,
                    on_tracker_started=on_tracker_started,
                )
            except Exception as error:
                rollback_error: Exception | None = None
                if rollback is not None and rollback_identity is not None and run_dir:
                    try:
                        _assert_run_directory_identity(
                            run_dir,
                            config.run.output_root,
                            rollback_identity,
                        )
                        _restore_resume_artifacts(run_dir, rollback)
                    except Exception as caught_rollback_error:
                        rollback_error = caught_rollback_error
                if rollback_error is not None:
                    raise ExceptionGroup(
                        "training failed and resume rollback also failed",
                        [error, rollback_error],
                    ) from error
                raise

            if tracker is None:
                raise RuntimeError("training completed without starting a tracker")
            complete_manifest = _load_json(result / "manifest.json")
            finish_summary = {
                "best_epoch": complete_manifest["best_epoch"],
                "best_metric": complete_manifest["best_metric"],
                "stop_reason": complete_manifest["stop_reason"],
            }
            finish_status = "success"
            return result
        finally:
            if tracker is not None:
                tracker.finish(finish_summary, status=finish_status)
