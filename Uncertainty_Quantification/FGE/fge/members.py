"""Strict readout auditing and weights-only A3 member artifacts."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from .errors import HardFailure


SCHEMA_VERSION = "upet.fge.member-delta.v1"
READOUT_PREFIXES = ("node_last_layers.", "edge_last_layers.")
READOUT_TENSOR_COUNT = 12
READOUT_SCALAR_COUNT = 13_338
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_PAYLOAD_KEYS = {
    "schema_version",
    "member_id",
    "cycle",
    "global_step",
    "base_sha256",
    "tensors",
}
_TENSOR_KEYS = {"name", "dtype", "shape", "value"}


@dataclass(frozen=True)
class ReadoutAudit:
    """Observable evidence for the formal UPET readout contract."""

    names: tuple[str, ...]
    tensor_count: int
    scalar_count: int
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[str, ...]
    all_finite: bool


@dataclass(frozen=True)
class TensorFingerprint:
    """Exact identity of one frozen parameter or buffer."""

    kind: str
    name: str
    dtype: str
    shape: tuple[int, ...]
    sha256: str


@dataclass(frozen=True)
class MemberTensor:
    """One validated full endpoint replacement."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    value: torch.Tensor


@dataclass(frozen=True)
class MemberPayload:
    """A validated weights-only A3 member."""

    schema_version: str
    member_id: int
    cycle: int
    global_step: int
    base_sha256: str
    tensors: tuple[MemberTensor, ...]


def _named_parameters(model: torch.nn.Module):
    return model.named_parameters(remove_duplicate=False)


def _is_readout(name: str) -> bool:
    return name.startswith(READOUT_PREFIXES)


def readout_tensor_names(model: torch.nn.Module) -> tuple[str, ...]:
    """Return readout parameter names in module registration order."""

    return tuple(name for name, _ in _named_parameters(model) if _is_readout(name))


def assert_readout_contract(model: torch.nn.Module) -> ReadoutAudit:
    """Require the exact CPU, finite, readout-only trainable scope."""

    named = list(_named_parameters(model))
    readout = [(name, parameter) for name, parameter in named if _is_readout(name)]
    names = tuple(name for name, _ in readout)
    if len(names) != len(set(names)) or len({id(value) for _, value in readout}) != len(
        readout
    ):
        raise HardFailure("readout contract contains duplicate tensors")
    if len(readout) != READOUT_TENSOR_COUNT:
        raise HardFailure(
            f"readout contract requires exactly 12 tensors, found {len(readout)}"
        )
    scalar_count = sum(parameter.numel() for _, parameter in readout)
    if scalar_count != READOUT_SCALAR_COUNT:
        raise HardFailure(
            f"readout contract requires exactly 13,338 scalars, found {scalar_count:,}"
        )
    unexpected = [
        name
        for name, parameter in named
        if not _is_readout(name) and parameter.requires_grad
    ]
    if unexpected:
        raise HardFailure(
            f"unexpected trainable parameters outside readout: {unexpected}"
        )
    if any(parameter.device.type != "cpu" for _, parameter in readout):
        raise HardFailure("readout tensors must be on CPU")
    if any(not parameter.is_floating_point() for _, parameter in readout):
        raise HardFailure("readout tensors must use floating dtypes")
    all_finite = all(bool(torch.isfinite(parameter).all()) for _, parameter in readout)
    if not all_finite:
        raise HardFailure("readout tensors must be finite")
    not_trainable = [name for name, parameter in readout if not parameter.requires_grad]
    if not_trainable:
        raise HardFailure(f"readout tensors are not trainable: {not_trainable}")
    return ReadoutAudit(
        names=names,
        tensor_count=len(readout),
        scalar_count=scalar_count,
        shapes=tuple(tuple(parameter.shape) for _, parameter in readout),
        dtypes=tuple(str(parameter.dtype) for _, parameter in readout),
        all_finite=all_finite,
    )


def _tensor_sha256(tensor: torch.Tensor) -> str:
    flat_bytes = (
        tensor.detach()
        .cpu()
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
        .numpy()
        .tobytes()
    )
    return hashlib.sha256(flat_bytes).hexdigest()


def frozen_fingerprint(model: torch.nn.Module) -> tuple[TensorFingerprint, ...]:
    """Fingerprint every non-readout parameter and every buffer exactly."""

    items = [
        ("parameter", name, tensor)
        for name, tensor in _named_parameters(model)
        if not _is_readout(name)
    ]
    items.extend(("buffer", name, tensor) for name, tensor in model.named_buffers())
    items.sort(key=lambda item: (item[1], item[0]))
    return tuple(
        TensorFingerprint(
            kind=kind,
            name=name,
            dtype=str(tensor.dtype),
            shape=tuple(tensor.shape),
            sha256=_tensor_sha256(tensor),
        )
        for kind, name, tensor in items
    )


def assert_frozen_unchanged(
    model: torch.nn.Module, expected: tuple[TensorFingerprint, ...]
) -> None:
    """Reject any exact metadata or content drift in frozen state."""

    if frozen_fingerprint(model) != expected:
        raise HardFailure("frozen state changed")


def _positive_int(value: object, field: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise HardFailure(f"{field} must be an integer")
    if value < (0 if allow_zero else 1):
        qualifier = "non-negative" if allow_zero else "positive"
        raise HardFailure(f"{field} must be {qualifier}")
    return value


def _base_sha(value: object, field: str = "base_sha256") -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise HardFailure(f"{field} must be a lowercase 64-hex SHA256")
    return value


def pack_member(
    model: torch.nn.Module,
    member_id: int,
    cycle: int,
    global_step: int,
    base_sha256: str,
) -> dict[str, object]:
    """Pack full endpoint replacements into the weights-only A3 schema."""

    member_id = _positive_int(member_id, "member_id")
    cycle = _positive_int(cycle, "cycle")
    global_step = _positive_int(global_step, "global_step", allow_zero=True)
    base_sha256 = _base_sha(base_sha256)
    audit = assert_readout_contract(model)
    parameters = dict(_named_parameters(model))
    tensors: list[dict[str, object]] = []
    for name in audit.names:
        value = parameters[name].detach().cpu().clone()
        tensors.append(
            {
                "name": name,
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "value": value,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "member_id": member_id,
        "cycle": cycle,
        "global_step": global_step,
        "base_sha256": base_sha256,
        "tensors": tensors,
    }


def _require_mapping(value: object, label: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise HardFailure(f"{label} must be a mapping")
    return value


def load_member(
    path: Path, base_sha256: str, expected_names: tuple[str, ...]
) -> MemberPayload:
    """Load and strictly validate a CPU A3 payload with safe deserialization."""

    expected_base = _base_sha(base_sha256, "expected base_sha256")
    raw_object = torch.load(path, weights_only=True, map_location="cpu")
    raw = _require_mapping(raw_object, "member payload")

    # Authenticate the base identity before inspecting replacement tensor content.
    actual_base = _base_sha(raw.get("base_sha256"))
    if actual_base != expected_base:
        raise HardFailure(
            f"base_sha256 mismatch: expected {expected_base}, found {actual_base}"
        )
    if set(raw) != _PAYLOAD_KEYS:
        raise HardFailure("member payload has missing or unexpected schema keys")
    if raw["schema_version"] != SCHEMA_VERSION:
        raise HardFailure("member payload schema_version mismatch")
    member_id = _positive_int(raw["member_id"], "member_id")
    cycle = _positive_int(raw["cycle"], "cycle")
    global_step = _positive_int(raw["global_step"], "global_step", allow_zero=True)

    if len(expected_names) != len(set(expected_names)):
        raise HardFailure("expected_names contains duplicates")
    entries_object = raw["tensors"]
    if not isinstance(entries_object, Sequence) or isinstance(
        entries_object, (str, bytes)
    ):
        raise HardFailure("member tensors must be a sequence")
    if len(entries_object) != len(expected_names):
        raise HardFailure("member tensor count does not match expected_names")

    tensors: list[MemberTensor] = []
    for expected_name, entry_object in zip(expected_names, entries_object, strict=True):
        entry = _require_mapping(entry_object, "member tensor entry")
        if set(entry) != _TENSOR_KEYS:
            raise HardFailure("member tensor has missing or unexpected schema keys")
        name = entry["name"]
        if not isinstance(name, str) or name != expected_name:
            raise HardFailure(
                "member tensor names or order do not match expected_names"
            )
        dtype = entry["dtype"]
        shape_object = entry["shape"]
        value = entry["value"]
        if not isinstance(dtype, str):
            raise HardFailure(f"member tensor {name} dtype metadata is invalid")
        if not isinstance(shape_object, (list, tuple)) or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 0
            for size in shape_object
        ):
            raise HardFailure(f"member tensor {name} shape metadata is invalid")
        shape = tuple(shape_object)
        if not isinstance(value, torch.Tensor):
            raise HardFailure(f"member tensor {name} value is not a tensor")
        if value.device.type != "cpu":
            raise HardFailure(f"member tensor {name} is not on CPU")
        if not value.is_floating_point():
            raise HardFailure(f"member tensor {name} is not floating point")
        if str(value.dtype) != dtype or tuple(value.shape) != shape:
            raise HardFailure(f"member tensor {name} dtype or shape metadata mismatch")
        if not bool(torch.isfinite(value).all()):
            raise HardFailure(f"member tensor {name} is not finite")
        tensors.append(MemberTensor(name, dtype, shape, value.detach().clone()))

    if tuple(tensor.name for tensor in tensors) != expected_names:
        raise HardFailure(
            "member contains missing, duplicate, extra, or reordered names"
        )
    return MemberPayload(
        schema_version=SCHEMA_VERSION,
        member_id=member_id,
        cycle=cycle,
        global_step=global_step,
        base_sha256=actual_base,
        tensors=tuple(tensors),
    )


def apply_member(
    model: torch.nn.Module,
    base_state: Mapping[str, torch.Tensor],
    member: MemberPayload,
) -> None:
    """Restore one immutable base snapshot, then apply endpoint replacements."""

    model_state = model.state_dict()
    if set(base_state) != set(model_state):
        raise HardFailure("base state names do not exactly match model state")
    for name, current in model_state.items():
        base = base_state[name]
        if not isinstance(base, torch.Tensor) or base.device.type != "cpu":
            raise HardFailure(f"base state {name} must be a CPU tensor")
        if base.dtype != current.dtype or tuple(base.shape) != tuple(current.shape):
            raise HardFailure(f"base state {name} dtype or shape mismatch")

    with torch.no_grad():
        for name, current in model_state.items():
            current.copy_(base_state[name])
    audit = assert_readout_contract(model)
    if tuple(tensor.name for tensor in member.tensors) != audit.names:
        raise HardFailure("member names do not exactly match model readout names")
    parameters = dict(_named_parameters(model))
    with torch.no_grad():
        for replacement in member.tensors:
            destination = parameters[replacement.name]
            if replacement.dtype != str(destination.dtype):
                raise HardFailure(f"member tensor {replacement.name} dtype mismatch")
            if replacement.shape != tuple(destination.shape):
                raise HardFailure(f"member tensor {replacement.name} shape mismatch")
            if replacement.value.device.type != "cpu":
                raise HardFailure(f"member tensor {replacement.name} must be on CPU")
            if replacement.value.dtype != destination.dtype:
                raise HardFailure(
                    f"member tensor {replacement.name} value dtype mismatch"
                )
            if tuple(replacement.value.shape) != tuple(destination.shape):
                raise HardFailure(
                    f"member tensor {replacement.name} value shape mismatch"
                )
            if not replacement.value.is_floating_point() or not bool(
                torch.isfinite(replacement.value).all()
            ):
                raise HardFailure(
                    f"member tensor {replacement.name} must be finite floating point"
                )
            destination.copy_(replacement.value)

    current_state = model.state_dict()
    for name, base in base_state.items():
        if not _is_readout(name) and not torch.equal(current_state[name].cpu(), base):
            raise HardFailure(
                f"frozen state {name} differs from base after member apply"
            )
