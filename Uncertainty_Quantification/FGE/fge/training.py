"""Native, source-independent FGE training orchestration."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch

from .artifacts import (
    ExperimentLayout,
    atomic_torch_save,
    atomic_write_json,
    sha256_file,
)
from .errors import HardFailure
from .manifests import build_training_manifest
from .members import (
    assert_frozen_unchanged,
    assert_readout_contract,
    frozen_fingerprint,
    pack_member,
)
from .schedule import asymmetric_triangular_lr


_LOSS_TERMS = (
    "energy",
    "forces",
    "virial",
    "non_conservative_forces",
    "non_conservative_stress",
)


@dataclass(frozen=True)
class TrainingBatchResult:
    """One differentiable batch loss and its recovered checkpoint loss terms."""

    loss: torch.Tensor
    loss_terms: tuple[str, ...]


@runtime_checkable
class TrainingRuntime(Protocol):
    """Backend boundary kept small enough for real-torch orchestration tests."""

    model: torch.nn.Module
    train_loader: Iterable[object]

    def train_batch(self, batch: object, *, lr: float) -> TrainingBatchResult:
        """Return the five-term differentiable loss for one collated batch."""

    def validate(self, *, use_ema: bool) -> Mapping[str, float]:
        """Evaluate raw or EMA readout weights without changing training state."""

    def assert_frozen(self) -> None:
        """Perform backend-specific frozen-state checks."""

    def reload_and_smoke(self, member_path: Path) -> bool:
        """Reload base+A3 and verify finite energy, force and stress outputs."""

    def resume_identity(self) -> Mapping[str, str]:
        """Return exact config/code/base/data identities for native resume."""

    def manifest_metadata(self) -> Mapping[str, Mapping[str, str]]:
        """Return formal dependency and code identities for publication."""


class _ReadoutEMA:
    """Raw-model preserving EMA that owns only currently trainable readout values."""

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        if not 0.0 <= decay < 1.0:
            raise HardFailure("EMA decay must lie in [0, 1)")
        self.decay = decay
        self.shadow = {
            name: value.detach().clone()
            for name, value in model.named_parameters()
            if value.requires_grad
        }
        if not self.shadow:
            raise HardFailure("EMA requires trainable readout tensors")

    def update(self, model: torch.nn.Module) -> None:
        parameters = dict(model.named_parameters())
        if set(parameters) < set(self.shadow):
            raise HardFailure("EMA trainable tensors changed")
        with torch.no_grad():
            for name, value in self.shadow.items():
                value.mul_(self.decay).add_(
                    parameters[name].detach(), alpha=1.0 - self.decay
                )

    @contextmanager
    def applied(self, model: torch.nn.Module):
        parameters = dict(model.named_parameters())
        raw = {name: parameters[name].detach().clone() for name in self.shadow}
        try:
            with torch.no_grad():
                for name, value in self.shadow.items():
                    parameters[name].copy_(value)
            yield
        finally:
            with torch.no_grad():
                for name, value in raw.items():
                    parameters[name].copy_(value)


def _layout(config: Any) -> ExperimentLayout:
    try:
        root = Path(config.paths.output_root) / config.project.name
    except AttributeError as exc:
        raise HardFailure("training configuration is incomplete") from exc
    if root.exists() and root.is_symlink():
        raise HardFailure("formal output root must not be a symlink")
    if root.exists() and (root / "result_manifest.json").exists():
        raise HardFailure("completed formal result is immutable")
    return ExperimentLayout(root)


def _loader_length(loader: Iterable[object]) -> int:
    try:
        length = len(loader)  # type: ignore[arg-type]
    except TypeError as exc:
        raise HardFailure("training loader must have a deterministic length") from exc
    if isinstance(length, bool) or not isinstance(length, int) or length < 1:
        raise HardFailure("training loader is empty")
    return length


def _resume_path(layout: ExperimentLayout) -> Path:
    return layout.root / "_work" / "native_resume.pt"


def _resume_identity(runtime: TrainingRuntime) -> dict[str, str]:
    identity = dict(runtime.resume_identity())
    if set(identity) != {"config", "code", "base", "data"} or any(
        not isinstance(value, str) or len(value) != 64 for value in identity.values()
    ):
        raise HardFailure("runtime resume identity is invalid")
    return identity


def _assert_resume_identity(
    layout: ExperimentLayout, runtime: TrainingRuntime, enabled: bool
) -> None:
    if not enabled:
        return
    path = _resume_path(layout)
    if not path.exists():
        return
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise HardFailure("native resume state cannot be read") from exc
    if not isinstance(state, Mapping) or state.get(
        "resume_identity"
    ) != _resume_identity(runtime):
        raise HardFailure("resume identity does not match config/code/base/data")


def _save_resume(
    layout: ExperimentLayout, runtime: TrainingRuntime, global_step: int
) -> None:
    atomic_torch_save(
        _resume_path(layout),
        {"resume_identity": _resume_identity(runtime), "global_step": global_step},
    )


def _assert_batch(result: object) -> TrainingBatchResult:
    if not isinstance(result, TrainingBatchResult):
        raise HardFailure("training runtime returned an invalid batch result")
    if not isinstance(result.loss, torch.Tensor) or result.loss.numel() != 1:
        raise HardFailure("training loss must be a scalar tensor")
    if not bool(torch.isfinite(result.loss).item()):
        raise HardFailure("training loss is not finite")
    if tuple(result.loss_terms) != _LOSS_TERMS:
        raise HardFailure("training loss does not contain the five-term contract")
    return result


def _assert_validation(values: Mapping[str, float]) -> None:
    if not values or any(
        not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in values.values()
    ):
        raise HardFailure("validation metrics are empty or non-finite")


def _canonical_sha256(value: object) -> str:
    """Hash a JSON-safe value with the one canonical FGE encoding."""

    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    except (TypeError, ValueError) as exc:
        raise HardFailure("manifest identity is not canonical JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _frozen_identity(frozen: Sequence[Any]) -> dict[str, str]:
    """Bind publication to the full ordered frozen model fingerprint."""

    records: list[dict[str, object]] = []
    for item in frozen:
        try:
            records.append(
                {
                    "kind": item.kind,
                    "name": item.name,
                    "dtype": item.dtype,
                    "shape": list(item.shape),
                    "sha256": item.sha256,
                }
            )
        except AttributeError as exc:
            raise HardFailure("frozen fingerprint has an invalid record") from exc
    return {"sha256": _canonical_sha256(records)}


def _manifest_identity(
    config: Any, runtime: TrainingRuntime, frozen: Sequence[Any]
) -> dict[str, object]:
    sanitized = getattr(config, "sanitized", None)
    if not callable(sanitized):
        raise HardFailure("training configuration has no sanitized identity")
    resolved = sanitized()
    try:
        identity = config.identity
        project = config.project
        scientific_flags = vars(config.scientific.training)
    except AttributeError as exc:
        raise HardFailure("training configuration identity is incomplete") from exc
    metadata = runtime.manifest_metadata()
    if set(metadata) != {
        "dependency_snapshot",
        "training_code_identity",
        "artifact_writer_code_identity",
        "validator_code_identity",
    }:
        raise HardFailure("runtime manifest metadata has an invalid schema")
    return {
        "project_name": project.name,
        "config_resolved": resolved,
        "config_identity": {"sha256": _canonical_sha256(resolved)},
        "checkpoint_identity": {"sha256": identity.base_checkpoint_sha256},
        "data_identities": {
            "train": {"sha256": identity.train_data_sha256},
            "val": {"sha256": identity.val_data_sha256},
            "test": {"sha256": identity.test_data_sha256},
        },
        "frozen_fingerprint_identity": _frozen_identity(frozen),
        "dependency_snapshot": metadata["dependency_snapshot"],
        "scientific_flags": scientific_flags,
        "training_code_identity": metadata["training_code_identity"],
        "artifact_writer_code_identity": metadata["artifact_writer_code_identity"],
        "validator_code_identity": metadata["validator_code_identity"],
    }


def train_fge(config: Any, *, runtime: TrainingRuntime | None = None) -> Path:
    """Train raw endpoint A3 members with one optimizer and validation-only EMA."""
    if runtime is None:
        raise HardFailure("native PET runtime is unavailable")
    if not isinstance(runtime, TrainingRuntime):
        raise HardFailure("training runtime does not implement the formal contract")
    layout = _layout(config)
    _assert_resume_identity(layout, runtime, bool(config.training.resume))
    audit = assert_readout_contract(runtime.model)
    frozen = frozen_fingerprint(runtime.model)
    trainable = [value for value in runtime.model.parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(
        trainable, lr=config.fge.lr_min, weight_decay=config.training.weight_decay
    )
    ema = _ReadoutEMA(runtime.model, config.ema.decay)
    steps_per_epoch = _loader_length(runtime.train_loader)
    updates_per_cycle = steps_per_epoch * config.fge.epochs_per_cycle
    if updates_per_cycle < 2:
        raise HardFailure("each FGE cycle requires at least two updates")

    global_step = 0
    accepted: list[dict[str, object]] = []
    for cycle in range(1, config.fge.cycles + 1):
        for epoch in range(config.fge.epochs_per_cycle):
            for step, batch in enumerate(runtime.train_loader):
                lr = asymmetric_triangular_lr(
                    epoch * steps_per_epoch + step,
                    updates_per_cycle,
                    config.fge.lr_min,
                    config.fge.lr_max,
                    config.fge.rise_fraction,
                )
                optimizer.zero_grad(set_to_none=True)
                result = _assert_batch(runtime.train_batch(batch, lr=lr))
                result.loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                ema.update(runtime.model)
                global_step += 1
            assert_frozen_unchanged(runtime.model, frozen)
            runtime.assert_frozen()
            _assert_validation(runtime.validate(use_ema=False))
            with ema.applied(runtime.model):
                _assert_validation(runtime.validate(use_ema=True))
            _save_resume(layout, runtime, global_step)

        assert_frozen_unchanged(runtime.model, frozen)
        runtime.assert_frozen()
        if any(not bool(torch.isfinite(value).all()) for value in trainable):
            raise HardFailure("endpoint readout tensors are not finite")
        member_path = layout.root / "training" / "members" / f"member_{cycle:03d}.pt"
        payload = pack_member(
            runtime.model,
            member_id=cycle,
            cycle=cycle,
            global_step=global_step,
            base_sha256=config.identity.base_checkpoint_sha256,
        )
        atomic_torch_save(member_path, payload)
        if not runtime.reload_and_smoke(member_path):
            member_path.unlink(missing_ok=True)
            raise HardFailure(f"member reload smoke failed: member_{cycle:03d}")
        accepted.append(
            {
                "member_id": f"member_{cycle:03d}",
                "cycle": cycle,
                "endpoint_global_step": global_step,
                "sha256": sha256_file(member_path),
            }
        )

    manifest = build_training_manifest(
        **_manifest_identity(config, runtime, frozen),
        model_contract={
            "readout_tensor_count": audit.tensor_count,
            "readout_parameter_count": audit.scalar_count,
        },
        member_count=len(accepted),
        members=accepted,
    )
    atomic_write_json(layout.training_manifest, manifest)
    return layout.training_manifest
