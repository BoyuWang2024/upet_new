"""Native, source-independent FGE training orchestration."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import secrets
import stat
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, TypedDict, runtime_checkable

import torch

from .artifacts import (
    ExperimentLayout,
    assert_safe_result_path,
    atomic_torch_save,
    atomic_write_json,
    sha256_file,
)
from .config import FGEConfig
from .errors import HardFailure
from .manifests import build_training_manifest
from .members import (
    apply_member,
    assert_frozen_unchanged,
    assert_readout_contract,
    frozen_fingerprint,
    load_member,
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


def _layout(config: FGEConfig) -> ExperimentLayout:
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


_WORK_DIRECTORY = ".fge_work"
_RESUME_NAME = "native_resume.pt"
_WORK_DIRECTORY_FLAGS = (
    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
)
_WORK_FILE_FLAGS = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)


def _assert_safe_project_component(config: FGEConfig) -> str:
    try:
        project = config.project.name
    except AttributeError as exc:
        raise HardFailure("training project is incomplete") from exc
    if (
        not isinstance(project, str)
        or not project
        or project in {".", ".."}
        or Path(project).name != project
        or "/" in project
        or "\\" in project
    ):
        raise HardFailure("training project name is not a safe path component")
    return project


_SAFE_WORK_PRIMITIVES = (
    all(operation in os.supports_dir_fd for operation in (os.open, os.mkdir, os.rename))
    and hasattr(os, "O_DIRECTORY")
    and hasattr(os, "O_NOFOLLOW")
)


def _assert_work_primitives() -> None:
    if not _SAFE_WORK_PRIMITIVES:
        raise HardFailure(
            "race-safe external work directory operations are unavailable"
        )


def _open_directory_component(parent_fd: int, component: str, *, create: bool) -> int:
    try:
        return os.open(component, _WORK_DIRECTORY_FLAGS, dir_fd=parent_fd)
    except FileNotFoundError:
        if not create:
            raise
        try:
            os.mkdir(component, 0o700, dir_fd=parent_fd)
        except FileExistsError:
            pass
        return os.open(component, _WORK_DIRECTORY_FLAGS, dir_fd=parent_fd)


def _open_output_root(config: FGEConfig, *, create: bool) -> int:
    _assert_work_primitives()
    try:
        output_root = Path(config.paths.output_root).absolute()
    except AttributeError as exc:
        raise HardFailure("training output root is incomplete") from exc
    if not output_root.is_absolute():
        raise HardFailure("training output root must be absolute")
    parts = output_root.parts
    if not parts or parts[0] != output_root.anchor:
        raise HardFailure("training output root has no trusted filesystem anchor")
    current = os.open(output_root.anchor, _WORK_DIRECTORY_FLAGS)
    try:
        for component in parts[1:]:
            child = _open_directory_component(current, component, create=create)
            os.close(current)
            current = child
        return current
    except BaseException:
        os.close(current)
        raise


def _open_work_directory(config: FGEConfig, *, create: bool) -> int:
    project = _assert_safe_project_component(config)
    output_fd: int | None = None
    work_fd: int | None = None
    try:
        output_fd = _open_output_root(config, create=create)
        work_fd = _open_directory_component(output_fd, _WORK_DIRECTORY, create=create)
        project_fd = _open_directory_component(work_fd, project, create=create)
        return project_fd
    except (FileNotFoundError, HardFailure):
        raise
    except OSError as exc:
        raise HardFailure("external FGE work directory is unsafe") from exc
    finally:
        if work_fd is not None:
            os.close(work_fd)
        if output_fd is not None:
            os.close(output_fd)


def _resume_identity(runtime: TrainingRuntime) -> dict[str, str]:
    identity = dict(runtime.resume_identity())
    if set(identity) != {"config", "code", "base", "data"} or any(
        not isinstance(value, str) or len(value) != 64 for value in identity.values()
    ):
        raise HardFailure("runtime resume identity is invalid")
    return identity


def _assert_resume_identity(
    config: FGEConfig, runtime: TrainingRuntime, enabled: bool
) -> None:
    if not enabled:
        return
    work_fd: int | None = None
    resume_fd: int | None = None
    try:
        work_fd = _open_work_directory(config, create=False)
        resume_fd = os.open(_RESUME_NAME, _WORK_FILE_FLAGS, dir_fd=work_fd)
        if not stat.S_ISREG(os.fstat(resume_fd).st_mode):
            raise HardFailure("native resume state is not a regular file")
        with os.fdopen(resume_fd, "rb", closefd=False) as handle:
            state = torch.load(handle, map_location="cpu", weights_only=True)
    except FileNotFoundError:
        return
    except HardFailure:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise HardFailure("native resume state cannot be read") from exc
    finally:
        if resume_fd is not None:
            os.close(resume_fd)
        if work_fd is not None:
            os.close(work_fd)
    if not isinstance(state, Mapping) or state.get(
        "resume_identity"
    ) != _resume_identity(runtime):
        raise HardFailure("resume identity does not match config/code/base/data")


def _save_resume(config: FGEConfig, runtime: TrainingRuntime, global_step: int) -> None:
    payload = io.BytesIO()
    torch.save(
        {"resume_identity": _resume_identity(runtime), "global_step": global_step},
        payload,
    )
    work_fd: int | None = None
    temporary_fd: int | None = None
    temporary_name = f".native_resume.{secrets.token_hex(16)}.tmp"
    try:
        work_fd = _open_work_directory(config, create=True)
        try:
            existing_fd = os.open(_RESUME_NAME, _WORK_FILE_FLAGS, dir_fd=work_fd)
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise HardFailure("external FGE resume path is unsafe") from exc
        else:
            try:
                if not stat.S_ISREG(os.fstat(existing_fd).st_mode):
                    raise HardFailure("external FGE resume path is unsafe")
            finally:
                os.close(existing_fd)
        temporary_fd = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=work_fd,
        )
        with os.fdopen(temporary_fd, "wb", closefd=False) as handle:
            handle.write(payload.getvalue())
            handle.flush()
            os.fsync(temporary_fd)
        os.rename(
            temporary_name,
            _RESUME_NAME,
            src_dir_fd=work_fd,
            dst_dir_fd=work_fd,
        )
        os.fsync(work_fd)
    except HardFailure:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise HardFailure("external FGE resume state cannot be written safely") from exc
    finally:
        if temporary_fd is not None:
            os.close(temporary_fd)
        if work_fd is not None:
            os.close(work_fd)


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


class _TrainingManifestIdentity(TypedDict):
    project_name: str
    config_resolved: Mapping[str, object]
    config_identity: Mapping[str, object]
    checkpoint_identity: Mapping[str, object]
    data_identities: Mapping[str, object]
    frozen_fingerprint_identity: Mapping[str, object]
    dependency_snapshot: Mapping[str, object]
    scientific_flags: Mapping[str, object]
    training_code_identity: Mapping[str, object]
    artifact_writer_code_identity: Mapping[str, object]
    validator_code_identity: Mapping[str, object]


def _manifest_identity(
    config: FGEConfig, runtime: TrainingRuntime, frozen: Sequence[Any]
) -> _TrainingManifestIdentity:
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


class PETTrainingRuntime:
    """Real PET restart runtime with frozen checkpoint transforms."""

    def __init__(
        self,
        model: torch.nn.Module,
        raw: Mapping[str, object],
        base_state: Mapping[str, torch.Tensor],
        loss_fn: Callable[[object, object, object], object],
        train_loader: Iterable[object],
        val_loader: Iterable[object],
        config: FGEConfig,
        unpack_batch: Any,
        batch_to: Any,
        evaluate_model: Any,
        average_by_num_atoms: Any,
        per_structure_targets: tuple[str, ...],
    ) -> None:
        self.model = model
        self._raw = raw
        self._base_state = {
            name: value.detach().cpu().clone() for name, value in base_state.items()
        }
        self._loss_fn = loss_fn
        self.train_loader = train_loader
        self._val_loader = val_loader
        self._config = config
        self._unpack_batch = unpack_batch
        self._batch_to = batch_to
        self._evaluate_model = evaluate_model
        self._average_by_num_atoms = average_by_num_atoms
        self._per_structure_targets = per_structure_targets
        self._frozen = frozen_fingerprint(model)

    @staticmethod
    def _targets(config: FGEConfig, path: Path) -> dict[str, object]:
        source = str(path)
        return {
            "energy": {
                "quantity": "energy",
                "read_from": source,
                "reader": "ase",
                "key": config.data.energy_target,
                "unit": "eV",
                "type": "scalar",
                "sample_kind": "system",
                "per_atom": False,
                "num_subtargets": 1,
                "forces": {"read_from": source, "key": "forces"},
                "stress": {"read_from": source, "key": "stress"},
                "virial": False,
            },
            "non_conservative_forces": {
                "quantity": "force",
                "read_from": source,
                "reader": "ase",
                "key": "forces",
                "unit": "eV/A",
                "type": {"cartesian": {"rank": 1}},
                "sample_kind": "atom",
                "per_atom": True,
                "num_subtargets": 1,
            },
            "non_conservative_stress": {
                "quantity": "pressure",
                "read_from": source,
                "reader": "ase",
                "key": "stress",
                "unit": "eV/A^3",
                "type": {"cartesian": {"rank": 2}},
                "sample_kind": "system",
                "per_atom": False,
                "num_subtargets": 1,
            },
        }

    @classmethod
    def from_config(cls, config: FGEConfig) -> "PETTrainingRuntime":
        """Construct only the official data/model/loss path; never refit transforms."""
        try:
            import copy
            import warnings

            from metatrain.utils.additive import get_remove_additive_transform
            from metatrain.utils.augmentation import RotationalAugmenter
            from metatrain.utils.data import (
                CollateFn,
                CombinedDataLoader,
                Dataset,
                read_systems,
                read_targets,
                unpack_batch,
            )
            from metatrain.utils.evaluate_model import evaluate_model
            from metatrain.utils.io import model_from_checkpoint
            from metatrain.utils.loss import LossAggregator
            from metatrain.utils.neighbor_lists import (
                get_requested_neighbor_lists,
                get_system_with_neighbor_lists_transform,
            )
            from metatrain.utils.per_atom import average_by_num_atoms
            from metatrain.utils.scaler import get_remove_scale_transform
            from metatrain.utils.transfer import batch_to
            from omegaconf import OmegaConf
            from torch.utils.data import DataLoader
        except ImportError as exc:
            raise HardFailure(
                "PET training runtime dependencies are unavailable"
            ) from exc
        if config.training.device != "cpu" or config.training.dtype != "float32":
            raise HardFailure("native FGE PET runtime requires CPU float32 readouts")
        raw = torch.load(
            config.paths.base_checkpoint, map_location="cpu", weights_only=False
        )
        if not isinstance(raw, Mapping):
            raise HardFailure("checkpoint is not a mapping")
        from .checkpoint import recover_loss_contract

        state = raw.get("model_state_dict")
        if (
            not isinstance(state, Mapping)
            or not state
            or any(
                not isinstance(name, str)
                or (not isinstance(value, torch.Tensor) and name != "finetune_config")
                for name, value in state.items()
            )
            or (
                "finetune_config" in state
                and not isinstance(state["finetune_config"], Mapping)
            )
        ):
            raise HardFailure("checkpoint is missing a valid restart model_state_dict")
        base_state = {
            name: value
            for name, value in state.items()
            if isinstance(value, torch.Tensor)
        }
        contract = recover_loss_contract(raw)
        model = model_from_checkpoint(raw, context="restart").to(
            device="cpu", dtype=torch.float32
        )
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(
                name.startswith(("node_last_layers.", "edge_last_layers."))
            )
        assert_readout_contract(model)

        for additive in model.additive_models:
            additive.to(dtype=torch.float64)
        model.scaler.to(dtype=torch.float64)
        model.additive_models[0].weights_to(device="cpu", dtype=torch.float64)
        additives = copy.deepcopy(
            model.additive_models.to(device="cpu", dtype=torch.float64)
        )
        model.additive_models[0].weights_to(device="cpu", dtype=torch.float64)
        model.scaler.scales_to(device="cpu", dtype=torch.float64)
        scaler = copy.deepcopy(model.scaler.to(device="cpu", dtype=torch.float64))
        model.scaler.scales_to(device="cpu", dtype=torch.float64)
        model.additive_models.to(device="cpu", dtype=torch.float32)
        model.additive_models[0].weights_to(device="cpu", dtype=torch.float32)
        model.scaler.to(device="cpu", dtype=torch.float32)
        model.scaler.scales_to(device="cpu", dtype=torch.float32)

        def build_dataset(path: Path):
            systems = read_systems(str(path))
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=(
                        r"the name of "
                        r"'non_conservative_(forces|stress)' resembles.*"
                    ),
                    category=UserWarning,
                )
                values, info = read_targets(
                    OmegaConf.create(cls._targets(config, path))
                )
            expected = model.dataset_info.targets
            if set(info) != set(expected) or any(
                not info[name].is_compatible_with(expected[name]) for name in expected
            ):
                raise HardFailure("dataset targets are incompatible with checkpoint")
            return Dataset.from_dict({"system": systems, **values})

        train_data, val_data = (
            build_dataset(config.paths.train_data),
            build_dataset(config.paths.val_data),
        )
        targets, extra = model.dataset_info.targets, model.dataset_info.extra_data
        fixed = [
            get_system_with_neighbor_lists_transform(
                get_requested_neighbor_lists(model)
            ),
            get_remove_additive_transform(additives, targets),
            get_remove_scale_transform(scaler),
        ]
        train_collate = CollateFn(
            target_keys=list(targets),
            callables=[
                RotationalAugmenter(
                    target_info_dict=targets, extra_data_info_dict=extra
                ).apply_random_augmentations,
                *fixed,
            ],
        )
        val_collate = CollateFn(target_keys=list(targets), callables=fixed)
        train_loader = CombinedDataLoader(
            [
                DataLoader(
                    train_data,
                    batch_size=config.training.batch_size,
                    shuffle=True,
                    drop_last=config.training.drop_last,
                    collate_fn=train_collate,
                    num_workers=config.training.num_workers,
                )
            ],
            shuffle=True,
        )
        val_loader = CombinedDataLoader(
            [
                DataLoader(
                    val_data,
                    batch_size=config.training.validation_batch_size,
                    shuffle=False,
                    drop_last=False,
                    collate_fn=val_collate,
                    num_workers=config.training.num_workers,
                )
            ],
            shuffle=False,
        )

        terms = {term.name: term for term in contract.terms}

        def loss_term(name: str) -> dict[str, object]:
            term = terms[name]
            return {
                "type": "huber",
                "delta": term.delta,
                "weight": term.weight,
                "reduction": contract.reduction,
                "gradients": {},
            }

        loss_config = {
            "energy": loss_term("energy"),
            "non_conservative_forces": loss_term("non_conservative_forces"),
            "non_conservative_stress": loss_term("non_conservative_stress"),
        }
        loss_config["energy"]["gradients"] = {
            "positions": loss_term("forces"),
            "strain": loss_term("virial"),
        }
        base_state = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
        return cls(
            model,
            raw,
            base_state,
            LossAggregator(targets=targets, config=loss_config),
            train_loader,
            val_loader,
            config,
            unpack_batch,
            batch_to,
            evaluate_model,
            average_by_num_atoms,
            contract.per_structure_targets,
        )

    def _loss(self, batch: object, training: bool) -> torch.Tensor:
        systems, targets, extra = self._unpack_batch(batch)
        systems, targets, extra = self._batch_to(
            systems, targets, extra, dtype=torch.float32, device=torch.device("cpu")
        )
        requested = {name: self.model.dataset_info.targets[name] for name in targets}
        predicted = self._evaluate_model(
            self.model, systems, requested, is_training=training
        )
        predicted = self._average_by_num_atoms(
            predicted, systems, self._per_structure_targets
        )
        targets = self._average_by_num_atoms(
            targets, systems, self._per_structure_targets
        )
        loss = self._loss_fn(predicted, targets, extra)
        if (
            not isinstance(loss, torch.Tensor)
            or loss.numel() != 1
            or not bool(torch.isfinite(loss).item())
        ):
            raise HardFailure("checkpoint-defined PET loss is invalid")
        return loss

    def train_batch(self, batch: object, *, lr: float) -> TrainingBatchResult:
        del lr
        self.model.train()
        return TrainingBatchResult(self._loss(batch, True), _LOSS_TERMS)

    def validate(self, *, use_ema: bool) -> Mapping[str, float]:
        del use_ema
        values: list[float] = []
        with torch.no_grad():
            for batch in self._val_loader:
                values.append(float(self._loss(batch, False).item()))
        if not values:
            raise HardFailure("validation loader is empty")
        return {"loss_total": sum(values) / len(values)}

    def assert_frozen(self) -> None:
        assert_frozen_unchanged(self.model, self._frozen)

    def reload_and_smoke(self, member_path: Path) -> bool:
        from metatrain.utils.io import model_from_checkpoint

        member = load_member(
            member_path,
            self._config.identity.base_checkpoint_sha256,
            assert_readout_contract(self.model),
        )
        model = model_from_checkpoint(self._raw, context="restart").to(
            device="cpu", dtype=torch.float32
        )
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(
                name.startswith(("node_last_layers.", "edge_last_layers."))
            )
        apply_member(model, self._base_state, member)
        try:
            batch = next(iter(self._val_loader))
            systems, targets, extra = self._unpack_batch(batch)
            systems, targets, _ = self._batch_to(
                systems, targets, extra, dtype=torch.float32, device=torch.device("cpu")
            )
            requested = {name: model.dataset_info.targets[name] for name in targets}
            with torch.no_grad():
                outputs = self._evaluate_model(
                    model, systems, requested, is_training=False
                )
            return all(
                bool(torch.isfinite(block.values).all())
                for value in outputs.values()
                for block in value.blocks()
            )
        except (RuntimeError, ValueError, OSError) as exc:
            raise HardFailure("member reload smoke execution failed") from exc

    def resume_identity(self) -> Mapping[str, str]:
        return {
            "config": _canonical_sha256(self._config.sanitized()),
            "code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "base": self._config.identity.base_checkpoint_sha256,
            "data": _canonical_sha256(
                {
                    "train": self._config.identity.train_data_sha256,
                    "val": self._config.identity.val_data_sha256,
                }
            ),
        }

    def manifest_metadata(self) -> Mapping[str, Mapping[str, str]]:
        return {
            "dependency_snapshot": {
                "torch": torch.__version__,
                "metatrain": "runtime",
            },
            "training_code_identity": {"status": "unavailable"},
            "artifact_writer_code_identity": {"status": "unavailable"},
            "validator_code_identity": {"status": "unavailable"},
        }


def train_fge(config: FGEConfig, *, runtime: TrainingRuntime | None = None) -> Path:
    """Train raw endpoint A3 members with one optimizer and validation-only EMA."""
    _assert_safe_project_component(config)
    _assert_work_primitives()
    if runtime is None:
        runtime = PETTrainingRuntime.from_config(config)
    if not isinstance(runtime, TrainingRuntime):
        raise HardFailure("training runtime does not implement the formal contract")
    layout = _layout(config)
    _assert_resume_identity(config, runtime, bool(config.training.resume))
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
                for group in optimizer.param_groups:
                    group["lr"] = lr
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
            _save_resume(config, runtime, global_step)

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
        assert_safe_result_path(layout.root, member_path)
        atomic_torch_save(member_path, payload)
        if not runtime.reload_and_smoke(member_path):
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
    assert_safe_result_path(layout.root, layout.training_manifest)
    atomic_write_json(layout.training_manifest, manifest)
    return layout.training_manifest
