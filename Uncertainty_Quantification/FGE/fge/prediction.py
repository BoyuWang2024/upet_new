"""Canonical CPU prediction payload validation and publication primitives."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import torch

from .artifacts import (
    ExperimentLayout,
    assert_safe_result_path,
    atomic_torch_save,
    atomic_write_json,
    sha256_file,
)
from .config import FGEConfig
from .data import DatasetIdentity
from .errors import HardFailure
from .manifests import build_prediction_manifest
from .members import ReadoutAudit, apply_member, assert_readout_contract, load_member


_PREDICTION_KEYS = frozenset(
    {
        "energy_prediction",
        "forces_prediction",
        "stress_prediction",
        "energy_reference",
        "forces_reference",
        "stress_reference",
        "n_atoms",
        "structure_offsets",
        "member_ids",
        "structure_ids",
        "atomic_numbers",
        "structure_mapping",
        "target_names",
        "units",
        "statistics",
    }
)
_OBSERVABLES = ("energy", "forces", "stress")


@dataclass(frozen=True)
class PredictionShape:
    """The three canonical prediction shape symbols."""

    K: int
    S: int
    A: int


def _tensor(
    payload: Mapping[str, object], name: str, shape: tuple[int, ...], dtype: torch.dtype
) -> torch.Tensor:
    value = payload.get(name)
    if not isinstance(value, torch.Tensor):
        raise HardFailure(f"{name} must be a tensor")
    if value.device.type != "cpu":
        raise HardFailure(f"{name} must be stored on CPU")
    if value.dtype != dtype:
        raise HardFailure(f"{name} has an invalid dtype")
    if tuple(value.shape) != shape:
        raise HardFailure(f"{name} has an invalid shape")
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise HardFailure(f"{name} must be finite")
    return value


def _ordered_ids(value: object, name: str, count: int | None) -> tuple[str, ...]:
    if (
        not isinstance(value, tuple)
        or (count is not None and len(value) != count)
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise HardFailure(f"{name} must be an ordered collection of unique IDs")
    return value


def _metadata(value: object, name: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(_OBSERVABLES):
        raise HardFailure(f"{name} must define energy, forces, and stress")
    result: dict[str, str] = {}
    for observable in _OBSERVABLES:
        item = value[observable]
        if not isinstance(item, str) or not item:
            raise HardFailure(f"{name}.{observable} must be a non-empty string")
        result[observable] = item
    return result


def validate_prediction_payload(payload: Mapping[str, object]) -> PredictionShape:
    """Require the complete, exact canonical prediction schema."""
    if not isinstance(payload, Mapping) or set(payload) != _PREDICTION_KEYS:
        raise HardFailure("canonical prediction has missing or unknown keys")

    energy_reference = payload["energy_reference"]
    forces_reference = payload["forces_reference"]
    if (
        not isinstance(energy_reference, torch.Tensor)
        or energy_reference.ndim != 1
        or not isinstance(forces_reference, torch.Tensor)
        or forces_reference.ndim != 2
        or forces_reference.shape[1:] != (3,)
    ):
        raise HardFailure("prediction references have invalid shapes")
    S = energy_reference.shape[0]
    A = forces_reference.shape[0]
    if S < 1 or A < 1:
        raise HardFailure("prediction shape symbols must be positive")

    member_ids = _ordered_ids(payload["member_ids"], "member_ids", None)
    K = len(member_ids)
    if K < 2:
        raise HardFailure("canonical prediction requires at least two members")
    expected_member_ids = tuple(f"member_{index:03d}" for index in range(1, K + 1))
    if member_ids != expected_member_ids:
        raise HardFailure("member_ids must be contiguous canonical FGE IDs")
    _ordered_ids(payload["structure_ids"], "structure_ids", S)

    _tensor(payload, "energy_prediction", (K, S), torch.float32)
    _tensor(payload, "forces_prediction", (K, A, 3), torch.float32)
    _tensor(payload, "stress_prediction", (K, S, 3, 3), torch.float32)
    _tensor(payload, "energy_reference", (S,), torch.float32)
    _tensor(payload, "forces_reference", (A, 3), torch.float32)
    _tensor(payload, "stress_reference", (S, 3, 3), torch.float32)
    n_atoms = _tensor(payload, "n_atoms", (S,), torch.int64)
    offsets = _tensor(payload, "structure_offsets", (S + 1,), torch.int64)
    atomic_numbers = _tensor(payload, "atomic_numbers", (A,), torch.int64)
    mapping = _tensor(payload, "structure_mapping", (A,), torch.int64)

    if bool((n_atoms <= 0).any()) or int(n_atoms.sum().item()) != A:
        raise HardFailure("n_atoms must describe every atom in non-empty structures")
    expected_offsets = torch.cat(
        (torch.zeros(1, dtype=torch.int64), n_atoms.cumsum(dim=0))
    )
    if not torch.equal(offsets, expected_offsets):
        raise HardFailure("structure_offsets are inconsistent with n_atoms")
    expected_mapping = torch.repeat_interleave(
        torch.arange(S, dtype=torch.int64), n_atoms
    )
    if not torch.equal(mapping, expected_mapping):
        raise HardFailure("structure_mapping is inconsistent with structure offsets")
    if bool((atomic_numbers < 1).any()) or bool((atomic_numbers > 118).any()):
        raise HardFailure("atomic_numbers must be valid positive atomic numbers")

    _metadata(payload["target_names"], "target_names")
    _metadata(payload["units"], "units")
    statistics = payload["statistics"]
    if (
        not isinstance(statistics, Mapping)
        or set(statistics) != {"K", "S", "A"}
        or any(
            isinstance(statistics[key], bool) or not isinstance(statistics[key], int)
            for key in ("K", "S", "A")
        )
        or (statistics["K"], statistics["S"], statistics["A"]) != (K, S, A)
    ):
        raise HardFailure("statistics must exactly match K, S, and A")
    return PredictionShape(K=K, S=S, A=A)


def canonical_prediction(payload: Mapping[str, object]) -> dict[str, object]:
    """Validate and detach a payload suitable for atomic tensor publication."""
    validate_prediction_payload(payload)
    result: dict[str, object] = {}
    for name, value in payload.items():
        if isinstance(value, torch.Tensor):
            result[name] = value.detach().cpu().clone()
        elif isinstance(value, tuple):
            result[name] = tuple(value)
        elif isinstance(value, Mapping):
            result[name] = dict(value)
        else:
            result[name] = value
    return result


class PredictionRuntime(Protocol):
    """Narrow dependency-injected seam around real base/A3 inference."""

    def load_base(self, config: FGEConfig) -> object: ...

    def restore_and_apply(self, base: object, member_id: str) -> None: ...

    def dataset_identity(self, config: FGEConfig) -> DatasetIdentity: ...

    def infer_member(
        self, base: object, member_id: str, config: FGEConfig
    ) -> Mapping[str, object]: ...


_MEMBER_OUTPUT_KEYS = frozenset(
    {
        "energy",
        "forces",
        "stress",
        "energy_reference",
        "forces_reference",
        "stress_reference",
        "n_atoms",
        "structure_offsets",
        "structure_ids",
        "atomic_numbers",
        "structure_mapping",
        "target_names",
        "units",
    }
)


def _training_manifest(path: Path) -> Mapping[str, object]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HardFailure("training manifest cannot be loaded") from exc
    if not isinstance(manifest, Mapping):
        raise HardFailure("training manifest has an invalid schema")
    return manifest


def _manifest_member_ids(path: Path, expected_count: int) -> tuple[str, ...]:
    manifest = _training_manifest(path)
    if not isinstance(manifest.get("members"), list):
        raise HardFailure("training manifest has no member list")
    member_ids: list[str] = []
    for member in cast(list[object], manifest["members"]):
        if not isinstance(member, Mapping) or not isinstance(
            member.get("member_id"), str
        ):
            raise HardFailure("training manifest has an invalid member entry")
        member_ids.append(member["member_id"])
    expected_ids = tuple(
        f"member_{index:03d}" for index in range(1, expected_count + 1)
    )
    if tuple(member_ids) != expected_ids:
        raise HardFailure("training manifest member count or order is invalid")
    return tuple(member_ids)


def _same_member_metadata(
    first: Mapping[str, object], current: Mapping[str, object]
) -> None:
    for name in _MEMBER_OUTPUT_KEYS - {"energy", "forces", "stress"}:
        left = first[name]
        right = current[name]
        if isinstance(left, torch.Tensor):
            if not isinstance(right, torch.Tensor) or not torch.equal(left, right):
                raise HardFailure(f"member inference metadata differs for {name}")
        elif left != right:
            raise HardFailure(f"member inference metadata differs for {name}")


def predict_members(config: object, *, runtime: object | None = None) -> Path:
    """Publish validated predictions in the exact training-manifest member order.

    ``runtime`` is the deliberately narrow inference seam: its real implementation
    loads the base and applies A3 members, while tests inject literal batch output.
    """
    if runtime is None:
        runtime = PETPredictionRuntime()
    typed_config = cast(FGEConfig, config)
    typed_runtime = cast(PredictionRuntime, runtime)
    try:
        project_name = typed_config.project.name
        output_root = Path(typed_config.paths.output_root)
        expected_count = typed_config.fge.member_count
        expected_test_data_sha256 = typed_config.identity.test_data_sha256
    except AttributeError as exc:
        raise HardFailure("predict_members requires an FGE configuration") from exc
    if (
        not isinstance(project_name, str)
        or not isinstance(expected_count, int)
        or not isinstance(expected_test_data_sha256, str)
    ):
        raise HardFailure("predict_members requires a valid FGE configuration")
    layout = ExperimentLayout(output_root / project_name)
    training_manifest = _training_manifest(layout.training_manifest)
    member_ids = _manifest_member_ids(layout.training_manifest, expected_count)
    load_base = getattr(runtime, "load_base", None)
    restore_and_apply = getattr(runtime, "restore_and_apply", None)
    infer_member = getattr(runtime, "infer_member", None)
    dataset_identity = getattr(runtime, "dataset_identity", None)
    if not callable(dataset_identity):
        raise HardFailure("inference runtime has an invalid seam")
    if not all(
        callable(method) for method in (load_base, restore_and_apply, infer_member)
    ):
        raise HardFailure("inference runtime has an invalid seam")

    expected_identity = dataset_identity(typed_config)
    if not isinstance(expected_identity, DatasetIdentity):
        raise HardFailure("inference runtime returned an invalid dataset identity")
    if expected_identity.content_sha256 != expected_test_data_sha256:
        raise HardFailure("runtime dataset identity differs from configured test data")
    expected_target_names = dict(expected_identity.target_names)
    expected_units = dict(expected_identity.units)
    base = typed_runtime.load_base(typed_config)
    member_outputs: list[Mapping[str, object]] = []
    for member_id in member_ids:
        typed_runtime.restore_and_apply(base, member_id)
        output = typed_runtime.infer_member(base, member_id, typed_config)
        if not isinstance(output, Mapping) or set(output) != _MEMBER_OUTPUT_KEYS:
            raise HardFailure("member inference output has an invalid schema")
        if output["structure_ids"] != expected_identity.structure_ids:
            raise HardFailure(
                "member inference structure IDs differ from dataset identity"
            )
        if output["target_names"] != expected_target_names:
            raise HardFailure(
                "member inference target names differ from dataset identity"
            )
        if output["units"] != expected_units:
            raise HardFailure("member inference units differ from dataset identity")
        if (
            not isinstance(output["energy_reference"], torch.Tensor)
            or not isinstance(output["forces_reference"], torch.Tensor)
            or output["energy_reference"].shape[0] != expected_identity.structure_count
            or output["forces_reference"].shape[0] != expected_identity.atom_count
        ):
            raise HardFailure("member inference shape differs from dataset identity")

        if member_outputs:
            _same_member_metadata(member_outputs[0], output)
        member_outputs.append(output)
    if len(member_outputs) != expected_count:
        raise HardFailure("member inference did not produce the required K members")

    first = member_outputs[0]
    payload = canonical_prediction(
        {
            "energy_prediction": torch.stack(
                [output["energy"] for output in member_outputs]
            ),
            "forces_prediction": torch.stack(
                [output["forces"] for output in member_outputs]
            ),
            "stress_prediction": torch.stack(
                [output["stress"] for output in member_outputs]
            ),
            "energy_reference": first["energy_reference"],
            "forces_reference": first["forces_reference"],
            "stress_reference": first["stress_reference"],
            "n_atoms": first["n_atoms"],
            "structure_offsets": first["structure_offsets"],
            "member_ids": member_ids,
            "structure_ids": first["structure_ids"],
            "atomic_numbers": first["atomic_numbers"],
            "structure_mapping": first["structure_mapping"],
            "target_names": first["target_names"],
            "units": first["units"],
            "statistics": {
                "K": expected_count,
                "S": cast(torch.Tensor, first["energy_reference"]).shape[0],
                "A": cast(torch.Tensor, first["forces_reference"]).shape[0],
            },
        }
    )
    assert_safe_result_path(layout.root, layout.prediction_tensor)
    atomic_torch_save(layout.prediction_tensor, payload)
    stored = torch.load(layout.prediction_tensor, weights_only=True, map_location="cpu")
    if not isinstance(stored, Mapping):
        raise HardFailure("published prediction is not a mapping")
    shape = validate_prediction_payload(stored)
    try:
        config_identity = training_manifest["config_identity"]
        data_identities = cast(
            Mapping[str, object], training_manifest["data_identities"]
        )
        test_data_identity = data_identities["test"]
        artifact_writer_code_identity = training_manifest[
            "artifact_writer_code_identity"
        ]
        validator_code_identity = training_manifest["validator_code_identity"]
    except (KeyError, TypeError) as exc:
        raise HardFailure(
            "training manifest lacks prediction publication identities"
        ) from exc
    manifest = build_prediction_manifest(
        root=layout.root,
        prediction_path=layout.prediction_tensor,
        member_ids=member_ids,
        shape={"K": shape.K, "S": shape.S, "A": shape.A},
        config_identity=cast(Mapping[str, object], config_identity),
        test_data_identity=cast(Mapping[str, object], test_data_identity),
        target_names=cast(Mapping[str, object], first["target_names"]),
        units=cast(Mapping[str, object], first["units"]),
        artifact_writer_code_identity=cast(
            Mapping[str, object], artifact_writer_code_identity
        ),
        validator_code_identity=cast(Mapping[str, object], validator_code_identity),
    )
    assert_safe_result_path(layout.root, layout.prediction_manifest)
    atomic_write_json(layout.prediction_manifest, manifest)
    return layout.prediction_tensor


def _ordered_ase_systems(systems: list[Any]) -> list[Any]:
    """Return extxyz structures in the canonical structure-ID order."""
    try:
        ordered = sorted(systems, key=lambda system: str(system.info["structure_id"]))
        identifiers = tuple(str(system.info["structure_id"]) for system in ordered)
    except (AttributeError, KeyError) as exc:
        raise HardFailure("PET prediction dataset lacks structure_id") from exc
    if len(set(identifiers)) != len(identifiers):
        raise HardFailure("PET prediction dataset has duplicate structure_id")
    return ordered


@dataclass
class PETBase:
    """One restart-restored PET model and immutable CPU base state."""

    model: torch.nn.Module
    state_dict: Mapping[str, torch.Tensor]
    audit: ReadoutAudit
    members_directory: Path
    base_sha256: str


@dataclass(frozen=True)
class PreparedPETChunk:
    """Neighbor-listed metatomic systems prepared once for all FGE members."""

    systems: tuple[Any, ...]


class PETPredictionRuntime:
    """Real metatrain PET implementation of the formal prediction seam."""

    def load_base(self, config: FGEConfig) -> PETBase:
        try:
            checkpoint_path = Path(config.paths.base_checkpoint)
            base_sha256 = config.identity.base_checkpoint_sha256
            members_directory = (
                Path(config.paths.output_root)
                / config.project.name
                / "training"
                / "members"
            )
        except AttributeError as exc:
            raise HardFailure("PET prediction configuration is incomplete") from exc
        return self.load_reused_base(
            checkpoint_path,
            base_sha256,
            members_directory,
        )

    def load_reused_base(
        self,
        base_checkpoint: str | Path,
        expected_sha256: str,
        members_directory: str | Path,
    ) -> PETBase:
        """Restore one authenticated base for a separately verified member tree."""
        checkpoint_path = Path(base_checkpoint)
        if sha256_file(checkpoint_path) != expected_sha256:
            raise HardFailure("base checkpoint SHA256 does not match configuration")
        try:
            import metatomic.torch  # noqa: F401
            from metatrain.utils.io import model_from_checkpoint

            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            if not isinstance(checkpoint, dict):
                raise HardFailure("PET restart checkpoint is not a mapping")
            model = model_from_checkpoint(checkpoint, context="restart")
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            raise HardFailure("unable to restore PET restart checkpoint") from exc
        model.to(device="cpu", dtype=torch.float32)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(
                name.startswith("node_last_layers.")
                or name.startswith("edge_last_layers.")
            )
        audit = assert_readout_contract(model)
        state_dict = {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        }
        return PETBase(
            model=model,
            state_dict=state_dict,
            audit=audit,
            members_directory=Path(members_directory),
            base_sha256=expected_sha256,
        )

    def restore_and_apply(self, base: object, member_id: str) -> None:
        if not isinstance(base, PETBase):
            raise HardFailure("PET prediction base is invalid")
        try:
            index = int(member_id.removeprefix("member_"))
        except ValueError as exc:
            raise HardFailure("PET prediction member ID is invalid") from exc
        if member_id != f"member_{index:03d}" or index < 1:
            raise HardFailure("PET prediction member ID is invalid")
        path = base.members_directory / f"{member_id}.pt"
        try:
            member = load_member(path, base.base_sha256, base.audit)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise HardFailure(f"unable to load PET member: {member_id}") from exc
        apply_member(base.model, base.state_dict, member)

    def dataset_identity(self, config: FGEConfig) -> DatasetIdentity:
        try:
            from ase.io import read

            systems = read(str(config.paths.test_data), ":")
        except (ImportError, OSError, ValueError) as exc:
            raise HardFailure("unable to read PET prediction dataset") from exc
        if not isinstance(systems, list) or not systems:
            raise HardFailure("PET prediction dataset is empty")
        systems = _ordered_ase_systems(cast(list[Any], systems))
        structure_ids = tuple(str(system.info["structure_id"]) for system in systems)
        return DatasetIdentity(
            split="test",
            structure_ids=structure_ids,
            structure_count=len(systems),
            atom_count=sum(len(system) for system in systems),
            content_sha256=sha256_file(config.paths.test_data),
            target_names=(
                ("energy", config.data.energy_target),
                ("forces", config.data.forces_target),
                ("stress", config.data.stress_target),
            ),
            units=(
                ("energy", config.data.energy_unit),
                ("forces", config.data.forces_unit),
                ("stress", config.data.stress_unit),
            ),
        )

    def prepare_ase_chunk(
        self,
        base: object,
        atoms: tuple[object, ...] | list[object],
    ) -> PreparedPETChunk:
        """Convert and neighbor-list one ASE chunk exactly once."""
        if not isinstance(base, PETBase):
            raise HardFailure("PET prediction base is invalid")
        if not atoms:
            raise HardFailure("PET prediction chunk is empty")
        try:
            from metatomic.torch import systems_to_torch
            from metatrain.utils.neighbor_lists import (
                get_requested_neighbor_lists,
                get_system_with_neighbor_lists,
            )

            model_systems = systems_to_torch(list(atoms), dtype=torch.float32)
            requested = get_requested_neighbor_lists(base.model)
            prepared = tuple(
                get_system_with_neighbor_lists(
                    system.to(dtype=torch.float32),
                    requested,
                )
                for system in model_systems
            )
        except (
            ImportError,
            AttributeError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            raise HardFailure("unable to prepare PET prediction chunk") from exc
        if len(prepared) != len(atoms):
            raise HardFailure("PET prediction chunk conversion changed its length")
        return PreparedPETChunk(systems=prepared)

    def infer_prepared(
        self,
        base: object,
        prepared: PreparedPETChunk,
    ) -> Mapping[str, torch.Tensor]:
        """Evaluate energy, force, and stress for one already-prepared chunk."""
        if not isinstance(base, PETBase) or not isinstance(prepared, PreparedPETChunk):
            raise HardFailure("PET prepared prediction inputs are invalid")
        if not prepared.systems:
            raise HardFailure("PET prepared prediction chunk is empty")
        try:
            from metatrain.utils.evaluate_model import evaluate_model

            targets = {
                name: base.model.dataset_info.targets[name]
                for name in (
                    "energy",
                    "non_conservative_forces",
                    "non_conservative_stress",
                )
            }
            base.model.eval()
            prediction = evaluate_model(
                base.model,
                list(prepared.systems),
                targets,
                is_training=False,
            )
            return {
                "energy": prediction["energy"]
                .block()
                .values.reshape(-1)
                .to(dtype=torch.float32, device="cpu"),
                "forces": prediction["non_conservative_forces"]
                .block()
                .values.squeeze(-1)
                .to(dtype=torch.float32, device="cpu"),
                "stress": prediction["non_conservative_stress"]
                .block()
                .values.squeeze(-1)
                .to(dtype=torch.float32, device="cpu"),
            }
        except (KeyError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            raise HardFailure("unable to evaluate PET prediction chunk") from exc

    def infer_member(
        self, base: object, member_id: str, config: FGEConfig
    ) -> Mapping[str, object]:
        """Infer one already-applied A3 member on the formal test split."""
        del member_id
        if not isinstance(base, PETBase):
            raise HardFailure("PET prediction base is invalid")
        try:
            from ase.io import read

            ase_systems = read(str(config.paths.test_data), ":")
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            raise HardFailure("unable to prepare PET prediction inputs") from exc
        if not isinstance(ase_systems, list) or not ase_systems:
            raise HardFailure("PET prediction dataset is empty")
        ase_systems = cast(list[Any], ase_systems)
        ase_systems = _ordered_ase_systems(ase_systems)
        prepared = self.prepare_ase_chunk(base, ase_systems)
        predictions = self.infer_prepared(base, prepared)
        energy = predictions["energy"]
        forces = predictions["forces"]
        stress = predictions["stress"]
        try:
            structure_ids = tuple(
                str(system.info["structure_id"]) for system in ase_systems
            )
            n_atoms = torch.tensor(
                [len(system) for system in ase_systems], dtype=torch.int64
            )
            structure_offsets = torch.cat(
                (torch.zeros(1, dtype=torch.int64), n_atoms.cumsum(dim=0))
            )
            atomic_numbers = torch.cat(
                [
                    torch.as_tensor(system.numbers, dtype=torch.int64)
                    for system in ase_systems
                ]
            )
            structure_mapping = torch.repeat_interleave(
                torch.arange(len(ase_systems), dtype=torch.int64), n_atoms
            )
            energy_reference = torch.tensor(
                [system.get_potential_energy() for system in ase_systems],
                dtype=torch.float32,
            )
            forces_reference = torch.cat(
                [
                    torch.as_tensor(system.get_forces(), dtype=torch.float32)
                    for system in ase_systems
                ]
            )
            stress_reference = torch.stack(
                [
                    torch.as_tensor(system.get_stress(voigt=False), dtype=torch.float32)
                    for system in ase_systems
                ]
            )
        except (KeyError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            raise HardFailure(
                "PET prediction dataset lacks required extxyz energy/forces/stress"
            ) from exc
        return {
            "energy": energy,
            "forces": forces,
            "stress": stress,
            "energy_reference": energy_reference,
            "forces_reference": forces_reference,
            "stress_reference": stress_reference,
            "n_atoms": n_atoms,
            "structure_offsets": structure_offsets,
            "structure_ids": structure_ids,
            "atomic_numbers": atomic_numbers,
            "structure_mapping": structure_mapping,
            "target_names": {
                "energy": config.data.energy_target,
                "forces": config.data.forces_target,
                "stress": config.data.stress_target,
            },
            "units": {
                "energy": config.data.energy_unit,
                "forces": config.data.forces_unit,
                "stress": config.data.stress_unit,
            },
        }
