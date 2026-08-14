"""Deterministic bootstrap member training and checkpoint publication."""

from __future__ import annotations

import copy
import random
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .artifacts import (
    atomic_replace_torch,
    atomic_write_json,
    atomic_write_npz,
    atomic_write_torch,
    sha256_file,
)
from .checkpoint import CHECKPOINT_SCHEMA, CheckpointAudit, audit_checkpoint
from .errors import HardFailure
from .head_policy import apply_pet_last_layer_policy
from .sampling import MemberSeeds, derive_member_seeds, draw_bootstrap_sample
from .training import EpochRecord, TrainingResume, TrainingRuntime, fit_runtime

RuntimeFactory = Callable[[Any, np.ndarray, MemberSeeds], TrainingRuntime]


def _state_subset(
    state: Mapping[str, torch.Tensor], names: tuple[str, ...]
) -> dict[str, torch.Tensor]:
    return {name: state[name].detach().cpu().clone() for name in names}


def _checkpoint(
    *,
    epoch: int,
    validation_loss: float,
    raw: dict[str, torch.Tensor],
    ema: dict[str, torch.Tensor],
    names: tuple[str, ...],
    resume: TrainingResume | None = None,
) -> dict[str, object]:
    document: dict[str, object] = {
        "schema": CHECKPOINT_SCHEMA,
        "epoch": epoch,
        "validation_loss": validation_loss,
        "raw_state_dict": raw,
        "ema_state_dict": ema,
    }
    if resume is not None:
        document.update(
            {
                "optimizer_state_dict": resume.optimizer_state,
                "python_rng_state": resume.python_rng_state,
                "numpy_rng_state": resume.numpy_rng_state,
                "torch_rng_state": resume.torch_rng_state,
                "runtime_state": resume.runtime_state,
                "best_epoch": resume.best_epoch,
                "best_loss": resume.best_loss,
                "best_raw_state_dict": _state_subset(resume.best_raw, names),
                "best_ema_state_dict": _state_subset(resume.best_ema, names),
                "history": [
                    {
                        "epoch": item.epoch,
                        "training_loss": item.training_loss,
                        "raw_validation_loss": item.raw_validation_loss,
                        "ema_validation_loss": item.ema_validation_loss,
                    }
                    for item in resume.history
                ],
            }
        )
    return document


def _full_state(
    runtime: TrainingRuntime,
    subset: Mapping[str, torch.Tensor],
    names: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    if set(subset) != set(names):
        raise HardFailure("resume checkpoint readout keys differ from PET policy")
    state = {
        name: value.detach().cpu().clone()
        for name, value in runtime.model.state_dict().items()
    }
    for name in names:
        value = subset[name]
        if not isinstance(value, torch.Tensor) or value.shape != state[name].shape:
            raise HardFailure(f"invalid resume tensor metadata: {name}")
        state[name] = value.detach().cpu().clone()
    return state


def _load_resume(
    path: Path, runtime: TrainingRuntime, names: tuple[str, ...]
) -> TrainingResume:
    try:
        document = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise HardFailure(
            f"could not load resume checkpoint {path}: {error}"
        ) from error
    if not isinstance(document, Mapping):
        raise HardFailure("resume checkpoint root must be a mapping")

    def branch(key: str) -> Mapping[str, torch.Tensor]:
        value = document.get(key)
        if not isinstance(value, Mapping) or any(
            not isinstance(name, str) or not isinstance(tensor, torch.Tensor)
            for name, tensor in value.items()
        ):
            raise HardFailure(f"resume checkpoint has invalid {key}")
        return value

    try:
        history = tuple(
            EpochRecord(
                epoch=int(item["epoch"]),
                training_loss=float(item["training_loss"]),
                raw_validation_loss=float(item["raw_validation_loss"]),
                ema_validation_loss=float(item["ema_validation_loss"]),
            )
            for item in document["history"]
        )
        optimizer = document["optimizer_state_dict"]
        runtime_state = document["runtime_state"]
        if not isinstance(optimizer, dict) or not isinstance(runtime_state, dict):
            raise TypeError("optimizer/runtime state")
        result = TrainingResume(
            completed_epoch=int(document["epoch"]),
            raw_state=_full_state(runtime, branch("raw_state_dict"), names),
            ema_state=_full_state(runtime, branch("ema_state_dict"), names),
            optimizer_state=copy.deepcopy(optimizer),
            best_epoch=int(document["best_epoch"]),
            best_loss=float(document["best_loss"]),
            best_raw=_full_state(runtime, branch("best_raw_state_dict"), names),
            best_ema=_full_state(runtime, branch("best_ema_state_dict"), names),
            history=history,
            python_rng_state=document["python_rng_state"],
            numpy_rng_state=document["numpy_rng_state"],
            torch_rng_state=document["torch_rng_state"],
            runtime_state=copy.deepcopy(runtime_state),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise HardFailure(f"resume checkpoint is incomplete: {error}") from error
    if not isinstance(result.numpy_rng_state, tuple) or not isinstance(
        result.torch_rng_state, torch.Tensor
    ):
        raise HardFailure("resume checkpoint RNG state is invalid")
    return result


def _checkpoint_record(path: Path, audit: CheckpointAudit) -> dict[str, object]:
    return {
        "path": f"checkpoints/{path.name}",
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "epoch": audit.epoch,
        "validation_loss": audit.validation_loss,
        "inference_ready": audit.inference_ready,
        "resume_ready": audit.resume_ready,
    }


def _publish_member_manifest(member_root: Path, index: int) -> Path:
    records: dict[str, object] = {}
    for kind in ("best", "final", "latest"):
        path = member_root / "checkpoints" / f"{kind}.pt"
        audit = audit_checkpoint(path, expected_parameter_count=13_338)
        records[kind] = _checkpoint_record(path, audit)
    return atomic_write_json(
        member_root / "manifest.json",
        {
            "schema": "upet.bootstrap.member/v1",
            "member_index": index,
            "checkpoints": records,
        },
    )


def _publish_sample(member_root: Path, sample: Any, config: Any) -> None:
    if not config.bootstrap.save_indices:
        return
    path = member_root / "bootstrap_indices.npz"
    arrays: dict[str, np.ndarray] = {"indices": sample.indices}
    if config.bootstrap.save_oob:
        arrays["oob"] = sample.oob
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as existing:
                if set(existing.files) != set(arrays) or any(
                    not np.array_equal(existing[name], value)
                    for name, value in arrays.items()
                ):
                    raise HardFailure("existing bootstrap indices differ")
        except (OSError, ValueError) as error:
            raise HardFailure(f"could not verify bootstrap indices: {error}") from error
        return
    atomic_write_npz(path, **arrays)


def train_run(
    config: Any,
    run_root: str | Path,
    *,
    training_size: int | None = None,
    runtime_factory: RuntimeFactory | None = None,
) -> Path:
    """Train configured members with resumable, canonical artifacts."""

    root = Path(run_root).expanduser().resolve()
    if runtime_factory is None:
        from .pet_training import PETTrainingRuntime

        runtime_factory = PETTrainingRuntime.from_config
    if training_size is None:
        try:
            from ase.io import read

            training_size = len(read(str(config.data.train), index=":"))
        except (ImportError, OSError, TypeError, ValueError) as error:
            raise HardFailure(
                f"could not determine training dataset size: {error}"
            ) from error
    member_paths: list[str] = []
    for index in range(config.bootstrap.ensemble_size):
        member_root = root / "members" / f"member_{index:03d}"
        checkpoint_root = member_root / "checkpoints"
        final_path = checkpoint_root / "final.pt"
        if final_path.exists():
            _publish_member_manifest(member_root, index)
            member_paths.append(f"members/member_{index:03d}/manifest.json")
            continue
        seeds = derive_member_seeds(config.bootstrap.base_seed, index)
        sample = draw_bootstrap_sample(
            training_size, seeds.sampling, config.bootstrap.sample_size
        )
        _publish_sample(member_root, sample, config)
        random.seed(seeds.python)
        np.random.seed(seeds.sampling)
        torch.manual_seed(seeds.torch)
        runtime = runtime_factory(config, sample.indices, seeds)
        policy = apply_pet_last_layer_policy(runtime.model)
        trainable_names = policy.trainable_names
        parameters = [
            parameter
            for parameter in runtime.model.parameters()
            if parameter.requires_grad
        ]
        if config.training.optimizer.name != "Adam":
            raise HardFailure("native bootstrap training currently requires Adam")
        optimizer = torch.optim.Adam(
            parameters,
            lr=config.training.optimizer.learning_rate,
            weight_decay=config.training.optimizer.weight_decay,
        )
        latest_path = checkpoint_root / "latest.pt"
        resume = (
            _load_resume(latest_path, runtime, trainable_names)
            if latest_path.exists()
            else None
        )

        def publish_latest(
            state: TrainingResume,
            names: tuple[str, ...] = trainable_names,
            destination: Path = latest_path,
            parameter_count: int = policy.trainable_parameter_count,
        ) -> None:
            loss = state.history[-1].raw_validation_loss
            document = _checkpoint(
                epoch=state.completed_epoch,
                validation_loss=loss,
                raw=_state_subset(state.raw_state, names),
                ema=_state_subset(state.ema_state, names),
                names=names,
                resume=state,
            )
            atomic_replace_torch(destination, document)
            audit = audit_checkpoint(
                destination,
                expected_parameter_count=parameter_count,
            )
            if not audit.resume_ready:
                raise HardFailure("published latest checkpoint is not resume ready")

        result = fit_runtime(
            runtime,
            optimizer,
            max_epochs=config.training.max_epochs,
            ema_decay=config.training.ema_decay,
            resume=resume,
            on_epoch=publish_latest,
        )
        if not latest_path.exists():
            publish_latest(result.resume)
        best_loss = next(
            item.raw_validation_loss
            for item in result.history
            if item.epoch == result.best_epoch
        )
        final_loss = result.history[-1].raw_validation_loss
        for kind, epoch, loss, raw, ema in (
            (
                "best",
                result.best_epoch,
                best_loss,
                result.best_raw,
                result.best_ema,
            ),
            (
                "final",
                result.resume.completed_epoch,
                final_loss,
                result.final_raw,
                result.final_ema,
            ),
        ):
            path = atomic_write_torch(
                checkpoint_root / f"{kind}.pt",
                _checkpoint(
                    epoch=epoch,
                    validation_loss=loss,
                    raw=_state_subset(raw, trainable_names),
                    ema=_state_subset(ema, trainable_names),
                    names=trainable_names,
                ),
            )
            audit_checkpoint(
                path, expected_parameter_count=policy.trainable_parameter_count
            )
        _publish_member_manifest(member_root, index)
        member_paths.append(f"members/member_{index:03d}/manifest.json")
    atomic_write_json(
        root / "run_manifest.json",
        {
            "schema": "upet.bootstrap.run/v1",
            "member_count": config.bootstrap.ensemble_size,
            "parameter_modes": list(config.prediction.parameter_modes),
            "stages": {
                "checkpoints": "complete",
                "predictions": "pending",
                "uncertainty": "pending",
            },
            "members": member_paths,
            "prediction_splits": [],
        },
    )
    return root
