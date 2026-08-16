"""Read-only authority for reusing one completed formal FGE ensemble."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from .artifacts import assert_safe_result_path, sha256_file
from .errors import HardFailure
from .inference_config import InferenceConfig
from .validation import validate_completed_result


_TRAINING_KEYS = {
    "schema_version",
    "project_name",
    "config_resolved",
    "config_identity",
    "checkpoint_identity",
    "data_identities",
    "model_contract",
    "frozen_fingerprint_identity",
    "dependency_snapshot",
    "scientific_flags",
    "training_code_identity",
    "artifact_writer_code_identity",
    "validator_code_identity",
    "member_count",
    "members",
}
_MEMBER_KEYS = {"member_id", "sha256", "cycle", "endpoint_global_step"}


@dataclass(frozen=True)
class EnsembleMember:
    member_id: str
    path: Path
    sha256: str
    cycle: int
    endpoint_global_step: int

    def canonical_identity(self) -> dict[str, object]:
        return {
            "member_id": self.member_id,
            "sha256": self.sha256,
            "cycle": self.cycle,
            "endpoint_global_step": self.endpoint_global_step,
        }


@dataclass(frozen=True)
class EnsembleAuthority:
    result_manifest_sha256: str
    training_manifest_sha256: str
    base_checkpoint_sha256: str
    members_directory: Path
    members: tuple[EnsembleMember, ...]
    artifact_writer_code_identity: Mapping[str, object]
    validator_code_identity: Mapping[str, object]

    @property
    def member_ids(self) -> tuple[str, ...]:
        return tuple(member.member_id for member in self.members)

    @property
    def member_sha256(self) -> tuple[str, ...]:
        return tuple(member.sha256 for member in self.members)

    def canonical_identity(self) -> dict[str, object]:
        """Return the reusable ensemble identity without filesystem locations."""
        return {
            "result_manifest_sha256": self.result_manifest_sha256,
            "training_manifest_sha256": self.training_manifest_sha256,
            "base_checkpoint_sha256": self.base_checkpoint_sha256,
            "members": [member.canonical_identity() for member in self.members],
            "artifact_writer_code_identity": dict(self.artifact_writer_code_identity),
            "validator_code_identity": dict(self.validator_code_identity),
        }


def _json_mapping(path: Path, label: str) -> Mapping[str, object]:
    if path.is_symlink() or not path.is_file():
        raise HardFailure(f"{label} must be a non-symlink regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise HardFailure(f"unable to load {label}") from exc
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise HardFailure(f"{label} has an invalid schema")
    return cast(Mapping[str, object], value)


def _identity_sha(value: object, label: str) -> str:
    if not isinstance(value, Mapping) or set(value) != {"sha256"}:
        raise HardFailure(f"{label} has an invalid schema")
    digest = value["sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise HardFailure(f"{label} has an invalid SHA-256")
    return digest


def _code_identity(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise HardFailure(f"{label} has an invalid schema")
    if dict(value) == {"status": "unavailable"}:
        return {"status": "unavailable"}
    if set(value) != {"commit", "dirty_sha256"}:
        raise HardFailure(f"{label} has an invalid schema")
    commit = value["commit"]
    dirty = value["dirty_sha256"]
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
        or not isinstance(dirty, str)
        or len(dirty) != 64
        or any(character not in "0123456789abcdef" for character in dirty)
    ):
        raise HardFailure(f"{label} has an invalid identity")
    return {"commit": commit, "dirty_sha256": dirty}


def open_ensemble_authority(config: InferenceConfig) -> EnsembleAuthority:
    """Reopen and bind every artifact needed to reuse a completed ensemble."""
    root = config.ensemble.root
    report = validate_completed_result(root)
    if report.status != "PASS" or report.mode != "read_only":
        raise HardFailure("completed ensemble validation did not remain read-only")

    result_path = root / "result_manifest.json"
    assert_safe_result_path(root, result_path)
    result_manifest_sha256 = sha256_file(result_path)
    if result_manifest_sha256 != config.ensemble.result_manifest_sha256:
        raise HardFailure("result manifest SHA-256 differs from inference config")

    training_path = root / "training" / "manifest.json"
    assert_safe_result_path(root, training_path)
    training = _json_mapping(training_path, "training manifest")
    if set(training) != _TRAINING_KEYS:
        raise HardFailure("training manifest has an invalid schema")
    if training.get("schema_version") != "upet.fge.training.v1":
        raise HardFailure("training manifest schema version is invalid")
    if training.get("member_count") != config.ensemble.member_count:
        raise HardFailure("training manifest member count differs from ensemble")
    base_sha256 = _identity_sha(
        training.get("checkpoint_identity"),
        "training checkpoint identity",
    )
    if base_sha256 != config.ensemble.base_checkpoint_sha256:
        raise HardFailure("training base checkpoint differs from inference config")

    raw_members = training.get("members")
    if not isinstance(raw_members, list) or len(raw_members) != 8:
        raise HardFailure("training members are invalid for reusable ensemble")
    members_directory = root / "training" / "members"
    assert_safe_result_path(root, members_directory / "member_001.pt")
    members: list[EnsembleMember] = []
    for index, raw_member in enumerate(raw_members, start=1):
        if not isinstance(raw_member, Mapping) or set(raw_member) != _MEMBER_KEYS:
            raise HardFailure("training member entry has an invalid schema")
        member_id = f"member_{index:03d}"
        digest = raw_member.get("sha256")
        cycle = raw_member.get("cycle")
        endpoint = raw_member.get("endpoint_global_step")
        if (
            raw_member.get("member_id") != member_id
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or type(cycle) is not int
            or cycle != index
            or type(endpoint) is not int
            or endpoint < 1
        ):
            raise HardFailure("training member identity is invalid")
        member_path = members_directory / f"{member_id}.pt"
        assert_safe_result_path(root, member_path)
        if member_path.is_symlink() or not member_path.is_file():
            raise HardFailure(f"member artifact is invalid: {member_id}")
        if sha256_file(member_path) != digest:
            raise HardFailure(f"member SHA-256 differs: {member_id}")
        members.append(
            EnsembleMember(
                member_id=member_id,
                path=member_path,
                sha256=digest,
                cycle=cycle,
                endpoint_global_step=endpoint,
            )
        )

    writer = _code_identity(
        training.get("artifact_writer_code_identity"),
        "artifact writer code identity",
    )
    validator = _code_identity(
        training.get("validator_code_identity"),
        "validator code identity",
    )
    return EnsembleAuthority(
        result_manifest_sha256=result_manifest_sha256,
        training_manifest_sha256=sha256_file(training_path),
        base_checkpoint_sha256=base_sha256,
        members_directory=members_directory,
        members=tuple(members),
        artifact_writer_code_identity=writer,
        validator_code_identity=validator,
    )


__all__ = ["EnsembleAuthority", "EnsembleMember", "open_ensemble_authority"]
