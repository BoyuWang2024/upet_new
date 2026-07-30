"""Train independent force and energy confidence readouts from a raw cache."""

from __future__ import annotations

import json
import os
import random
import tempfile
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from ..artifacts import atomic_write_json, sha256_file
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
)


RUN_SCHEMA_VERSION = "upet_confidence_run_v1"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON {path} must contain a mapping")
    return value


def _safe_run_dir(root: Path, run_name: str) -> Path:
    if (
        not isinstance(run_name, str)
        or not run_name
        or run_name in {".", ".."}
        or Path(run_name).name != run_name
    ):
        raise ValueError(f"unsafe run name: {run_name!r}")
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


def train_run(
    config: ConfidenceConfig,
    *,
    cache_manifest_path: Path,
    run_name: str,
    stop_after_epoch: int | None = None,
) -> Path:
    """Train confidence heads and atomically publish a complete run manifest."""
    if config.binning.force_num_bins != config.binning.energy_num_bins:
        raise ValueError("force and energy num_bins must match for ConfidenceModel")
    cache_path = Path(cache_manifest_path).resolve()
    cache_manifest = _load_json(cache_path)
    if cache_manifest.get("status") != "complete" or not isinstance(
        cache_manifest.get("cache_id"), str
    ):
        raise ValueError("cache manifest must be complete and identified")
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
    run_dir = _safe_run_dir(config.run.output_root, run_name)
    run_dir.mkdir(parents=True)
    manifest_path = run_dir / "manifest.json"
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
        "artifacts": {},
    }
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
        num_bins=config.binning.force_num_bins,
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
    externally_stopped = False
    for epoch in range(config.trainer.max_epochs):
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
            "external_stop_after_epoch" if externally_stopped else control.stop_reason
        ),
        "epochs_completed": len(metric_records),
        "force_feature_dim": force_dim,
        "energy_feature_dim": energy_dim,
    }
    atomic_write_json(manifest_path, complete)
    return run_dir
