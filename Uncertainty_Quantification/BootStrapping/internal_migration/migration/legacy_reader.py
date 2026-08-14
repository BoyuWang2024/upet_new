"""Read-only validation of the previously established result schema."""

from __future__ import annotations

import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from ...bootstrap.errors import HardFailure


_CHUNK_PATTERN = re.compile(r"chunk_([0-9]{5})[.]pt\Z")


@dataclass(frozen=True)
class LegacyChunk:
    path: Path
    member_index: int
    member_id: str
    split: str
    chunk_index: int
    structure_start: int
    structure_count: int


@dataclass(frozen=True)
class LegacyMember:
    member_index: int
    member_id: str
    checkpoints: dict[str, Path]
    chunks: dict[str, tuple[LegacyChunk, ...]]


@dataclass(frozen=True)
class LegacyRunAudit:
    source: Path
    members: tuple[LegacyMember, ...]
    splits: tuple[str, ...]


def load_legacy_payload(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        raise HardFailure(f"could not load existing result {path}: {error}") from error
    if not isinstance(payload, dict):
        raise HardFailure(f"existing result root must be a mapping: {path}")
    return payload


def _regular(path: Path, location: str) -> None:
    if path.is_symlink():
        raise HardFailure(f"{location} must not be a symlink: {path}")
    try:
        mode = path.stat().st_mode
    except OSError as error:
        raise HardFailure(f"missing {location}: {path}") from error
    if not stat.S_ISREG(mode):
        raise HardFailure(f"{location} must be a regular file: {path}")


def _identity(payload: Mapping[str, Any], path: Path) -> Mapping[str, Any]:
    identity = payload.get("identity")
    if not isinstance(identity, Mapping):
        raise HardFailure(f"prediction chunk identity is missing: {path}")
    return identity


def _read_chunks(
    source: Path, member_index: int, member_id: str, split: str
) -> tuple[LegacyChunk, ...]:
    root = source / "predictions" / split / member_id
    try:
        paths = sorted(path for path in root.iterdir() if path.is_file())
    except OSError as error:
        raise HardFailure(f"prediction chunk directory is missing: {root}") from error
    chunks: list[LegacyChunk] = []
    expected_start = 0
    for expected_index, path in enumerate(paths):
        match = _CHUNK_PATTERN.fullmatch(path.name)
        if match is None or int(match.group(1)) != expected_index:
            raise HardFailure(f"prediction chunk sequence is not contiguous: {root}")
        _regular(path, "prediction chunk")
        payload = load_legacy_payload(path)
        identity = _identity(payload, path)
        expected = {
            "split": split,
            "branch": "raw",
            "member_id": member_id,
            "chunk_index": expected_index,
            "structure_start": expected_start,
        }
        if any(identity.get(key) != value for key, value in expected.items()):
            raise HardFailure(f"prediction chunk identity mismatch: {path}")
        count = identity.get("structure_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise HardFailure(f"prediction chunk structure_count is invalid: {path}")
        chunks.append(
            LegacyChunk(
                path=path,
                member_index=member_index,
                member_id=member_id,
                split=split,
                chunk_index=expected_index,
                structure_start=expected_start,
                structure_count=count,
            )
        )
        expected_start += count
    if not chunks:
        raise HardFailure(f"prediction chunk sequence is empty: {root}")
    return tuple(chunks)


def inspect_legacy_run(source: str | Path, config: object) -> LegacyRunAudit:
    """Audit configured members and contiguous chunks without computing outputs."""

    source_path = Path(source).expanduser().resolve()
    if not source_path.is_dir() or source_path.is_symlink():
        raise HardFailure(f"existing run root is invalid: {source_path}")
    try:
        member_count = config.bootstrap.ensemble_size  # type: ignore[attr-defined]
        splits = tuple(config.prediction.splits)  # type: ignore[attr-defined]
        modes = tuple(config.prediction.parameter_modes)  # type: ignore[attr-defined]
    except AttributeError as error:
        raise HardFailure("adapter configuration is incomplete") from error
    if modes != ("raw",):
        raise HardFailure("existing result adapter accepts raw predictions only")
    members: list[LegacyMember] = []
    reference_layout: dict[str, tuple[tuple[int, int], ...]] = {}
    for zero_index in range(member_count):
        old_index = zero_index + 1
        member_id = f"member_{old_index:02d}"
        member_root = source_path / member_id
        checkpoints = {
            kind: member_root / f"{kind}.pt" for kind in ("best", "final", "latest")
        }
        for kind, path in checkpoints.items():
            _regular(path, f"{kind} checkpoint")
        chunks = {
            split: _read_chunks(source_path, old_index, member_id, split)
            for split in splits
        }
        for split, split_chunks in chunks.items():
            layout = tuple(
                (chunk.structure_start, chunk.structure_count) for chunk in split_chunks
            )
            if split in reference_layout and reference_layout[split] != layout:
                raise HardFailure(
                    f"prediction chunk layout differs across members: {split}"
                )
            reference_layout.setdefault(split, layout)
        members.append(LegacyMember(old_index, member_id, checkpoints, chunks))
    return LegacyRunAudit(source_path, tuple(members), splits)
