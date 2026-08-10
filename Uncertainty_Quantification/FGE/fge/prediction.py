"""Canonical CPU prediction payload validation and publication primitives."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import torch

from .artifacts import ExperimentLayout, atomic_torch_save
from .config import FGEConfig
from .errors import HardFailure


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
    structure_ids = _ordered_ids(payload["structure_ids"], "structure_ids", S)
    if tuple(sorted(structure_ids)) != structure_ids:
        raise HardFailure("structure_ids must be in canonical order")

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


def _manifest_member_ids(path: Path, expected_count: int) -> tuple[str, ...]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HardFailure("training manifest cannot be loaded") from exc
    if not isinstance(manifest, Mapping) or not isinstance(
        manifest.get("members"), list
    ):
        raise HardFailure("training manifest has no member list")
    member_ids: list[str] = []
    for member in manifest["members"]:
        if not isinstance(member, Mapping) or not isinstance(
            member.get("member_id"), str
        ):
            raise HardFailure("training manifest has an invalid member entry")
        member_ids.append(member["member_id"])
    if len(member_ids) != expected_count or len(set(member_ids)) != len(member_ids):
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
        raise HardFailure("predict_members requires an inference runtime")
    typed_config = cast(FGEConfig, config)
    typed_runtime = cast(PredictionRuntime, runtime)
    try:
        project_name = typed_config.project.name
        output_root = Path(typed_config.paths.output_root)
        expected_count = typed_config.fge.member_count
    except AttributeError as exc:
        raise HardFailure("predict_members requires an FGE configuration") from exc
    if not isinstance(project_name, str) or not isinstance(expected_count, int):
        raise HardFailure("predict_members requires a valid FGE configuration")
    layout = ExperimentLayout(output_root / project_name)
    member_ids = _manifest_member_ids(layout.training_manifest, expected_count)
    load_base = getattr(runtime, "load_base", None)
    restore_and_apply = getattr(runtime, "restore_and_apply", None)
    infer_member = getattr(runtime, "infer_member", None)
    if not all(
        callable(method) for method in (load_base, restore_and_apply, infer_member)
    ):
        raise HardFailure("inference runtime has an invalid seam")

    base = typed_runtime.load_base(typed_config)
    member_outputs: list[Mapping[str, object]] = []
    for member_id in member_ids:
        typed_runtime.restore_and_apply(base, member_id)
        output = typed_runtime.infer_member(base, member_id, typed_config)
        if not isinstance(output, Mapping) or set(output) != _MEMBER_OUTPUT_KEYS:
            raise HardFailure("member inference output has an invalid schema")
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
    atomic_torch_save(layout.prediction_tensor, payload)
    stored = torch.load(layout.prediction_tensor, weights_only=True, map_location="cpu")
    if not isinstance(stored, Mapping):
        raise HardFailure("published prediction is not a mapping")
    validate_prediction_payload(stored)
    return layout.prediction_tensor
