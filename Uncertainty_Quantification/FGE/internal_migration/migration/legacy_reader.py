"""Read one explicitly identified successful legacy FGE result without mutation."""

from __future__ import annotations

import csv
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ...fge.artifacts import sha256_file
from ...fge.errors import HardFailure
from ...fge.prediction import canonical_prediction


_SHA_RE = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class LegacyExpectations:
    """Explicit identities and layout for the one accepted source run."""

    member_count: int
    chunk_count: int
    member_sha256: Mapping[str, str]
    test_data_sha256: str
    manifest_path: str
    members_directory: str
    prediction_directory: str
    uncertainty_path: str
    metrics_path: str
    risk_coverage_paths: Mapping[str, str] | None = None


@dataclass(frozen=True)
class LegacyMember:
    """One accepted legacy endpoint and its immutable source identity."""

    member_id: str
    cycle: int
    global_step: int
    checkpoint_path: Path
    sha256: str


@dataclass(frozen=True)
class SourceSnapshot:
    """Critical source files and hashes captured before deserialization."""

    root: Path
    files: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class LegacyRun:
    """Validated legacy values plus audit-only source provenance."""

    members: tuple[LegacyMember, ...]
    prediction: dict[str, object]
    legacy_uncertainty: Mapping[str, object]
    legacy_metrics: Mapping[str, object]
    legacy_risk_coverage: Mapping[str, tuple[Mapping[str, object], ...]]
    source_snapshot: SourceSnapshot


class _SnapshotBuilder:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.files: dict[str, str] = {}

    def record(self, path: Path) -> None:
        relative = path.relative_to(self.root).as_posix()
        if relative in self.files:
            return
        self.files[relative] = sha256_file(path)

    def finish(self) -> SourceSnapshot:
        snapshot = SourceSnapshot(self.root, tuple(sorted(self.files.items())))
        verify_source_unchanged(snapshot)
        return snapshot


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise HardFailure(f"{label} must be a positive integer")
    return value


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise HardFailure(f"{label} must be a lowercase SHA256")
    return value


def _source_root(root: Path) -> Path:
    source = Path(root)
    if not source.is_dir() or source.is_symlink():
        raise HardFailure("legacy source root must be a non-symlink directory")
    return source.resolve()


def _inside(root: Path, value: str | Path, label: str) -> Path:
    raw = Path(value)
    candidate = raw if raw.is_absolute() else root / raw
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise HardFailure(f"{label} escapes or is missing from legacy source") from exc
    if candidate.is_symlink() or not resolved.is_file():
        raise HardFailure(f"{label} must be a non-symlink regular file")
    ancestor = candidate
    while ancestor != root:
        if ancestor.is_symlink():
            raise HardFailure(f"{label} traverses a symlink")
        ancestor = ancestor.parent
    return resolved


def _directory(root: Path, value: str | Path, label: str) -> Path:
    raw = Path(value)
    candidate = raw if raw.is_absolute() else root / raw
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise HardFailure(f"{label} escapes or is missing from legacy source") from exc
    if candidate.is_symlink() or not resolved.is_dir():
        raise HardFailure(f"{label} must be a non-symlink directory")
    ancestor = candidate
    while ancestor != root:
        if ancestor.is_symlink():
            raise HardFailure(f"{label} traverses a symlink")
        ancestor = ancestor.parent
    return resolved


def _json(path: Path, snapshot: _SnapshotBuilder) -> dict[str, Any]:
    snapshot.record(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HardFailure(f"invalid legacy JSON: {path.name}") from exc
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise HardFailure(f"legacy JSON must be a string-keyed mapping: {path.name}")
    return value


def _torch(path: Path, snapshot: _SnapshotBuilder) -> Mapping[str, object]:
    snapshot.record(path)
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        raise HardFailure(f"invalid legacy tensor artifact: {path.name}") from exc
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise HardFailure(f"legacy tensor artifact is not a mapping: {path.name}")
    return value


def _tensor(
    payload: Mapping[str, object],
    key: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    value = payload.get(key)
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.dtype != dtype
        or tuple(value.shape) != shape
        or (value.is_floating_point() and not bool(torch.isfinite(value).all()))
    ):
        raise HardFailure(f"legacy chunk {key} has invalid dtype, shape, or values")
    return value


def _expectation_schema(expected: LegacyExpectations) -> tuple[str, ...]:
    K = _positive_int(expected.member_count, "member_count")
    _positive_int(expected.chunk_count, "chunk_count")
    _sha(expected.test_data_sha256, "test_data_sha256")
    ids = tuple(f"member_{index:03d}" for index in range(1, K + 1))
    if set(expected.member_sha256) != set(ids):
        raise HardFailure("expected member SHA identities are incomplete")
    for member_id in ids:
        _sha(expected.member_sha256[member_id], f"{member_id} SHA256")
    return ids


def _members(
    root: Path,
    manifest: Mapping[str, object],
    expected: LegacyExpectations,
    member_ids: tuple[str, ...],
    snapshot: _SnapshotBuilder,
) -> tuple[LegacyMember, ...]:
    entries = manifest.get("members")
    if not isinstance(entries, list) or len(entries) != len(member_ids):
        raise HardFailure("legacy manifest member count is invalid")
    member_root = _directory(root, expected.members_directory, "members directory")
    result: list[LegacyMember] = []
    for index, (member_id, raw) in enumerate(
        zip(member_ids, entries, strict=False), start=1
    ):
        if not isinstance(raw, Mapping) or raw.get("member_id") != member_id:
            raise HardFailure("legacy manifest member order is invalid")
        if not (
            raw.get("accepted") is True
            and raw.get("status") == "valid"
            and raw.get("reload_check") == "passed"
            and raw.get("finite_check") == "passed"
            and type(raw.get("frozen_drift")) in (int, float)
            and float(raw["frozen_drift"]) == 0.0
        ):
            raise HardFailure(f"legacy member is not an accepted success: {member_id}")
        cycle = _positive_int(raw.get("cycle_index"), "legacy cycle_index")
        if cycle != index:
            raise HardFailure("legacy member cycles are not contiguous")
        step = _positive_int(
            raw.get("global_step_after_optimizer"), "legacy global step"
        )
        path = _inside(root, raw.get("checkpoint_path", ""), "member checkpoint")
        expected_path = member_root / f"{member_id}.ckpt"
        if path != expected_path.resolve():
            raise HardFailure("legacy member checkpoint path is not canonical")
        snapshot.record(path)
        digest = sha256_file(path)
        if digest != expected.member_sha256[member_id]:
            raise HardFailure(f"legacy member SHA256 mismatch: {member_id}")
        result.append(LegacyMember(member_id, cycle, step, path, digest))
    return tuple(result)


def _prediction(
    root: Path,
    expected: LegacyExpectations,
    member_ids: tuple[str, ...],
    snapshot: _SnapshotBuilder,
) -> dict[str, object]:
    prediction_root = _directory(
        root, expected.prediction_directory, "prediction directory"
    )
    summary_path = _inside(
        root, prediction_root / "prediction_summary.json", "prediction summary"
    )
    summary = _json(summary_path, snapshot)
    if summary.get("split") != "test" or summary.get("inference_only") is not True:
        raise HardFailure(
            "legacy prediction is not the successful inference test split"
        )
    if tuple(summary.get("member_ids", ())) != member_ids:
        raise HardFailure("legacy prediction member order is invalid")
    identity = summary.get("dataset_identity")
    if (
        not isinstance(identity, Mapping)
        or identity.get("sha256") != expected.test_data_sha256
    ):
        raise HardFailure("legacy test dataset identity is invalid")
    S = _positive_int(summary.get("n_structures"), "legacy structure count")
    n_atoms_summary = tuple(summary.get("n_atoms", ()))
    structure_ids_raw = tuple(summary.get("structure_ids", ()))
    if (
        len(n_atoms_summary) != S
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in n_atoms_summary
        )
        or len(structure_ids_raw) != S
    ):
        raise HardFailure("legacy prediction summary mapping is invalid")
    structure_ids = tuple(str(value) for value in structure_ids_raw)
    if len(set(structure_ids)) != S:
        raise HardFailure("legacy structure IDs are not unique")
    chunks = summary.get("chunks")
    listed = summary.get("member_chunks")
    if (
        not isinstance(chunks, list)
        or len(chunks) != expected.chunk_count
        or not isinstance(listed, Mapping)
    ):
        raise HardFailure("legacy chunk inventory is invalid")
    next_structure = 0
    chunk_specs: list[tuple[int, int]] = []
    for index, raw in enumerate(chunks):
        if not isinstance(raw, Mapping) or raw.get("chunk_index") != index:
            raise HardFailure("legacy chunk indices are not contiguous")
        start = raw.get("structure_start")
        count = raw.get("structure_count")
        if start != next_structure:
            raise HardFailure("legacy chunk structure ranges are not contiguous")
        count = _positive_int(count, "legacy chunk structure count")
        next_structure += count
        chunk_specs.append((start, count))
    if next_structure != S:
        raise HardFailure("legacy chunks do not cover all structures")

    predictions: dict[str, list[torch.Tensor]] = {
        "energy": [],
        "forces": [],
        "stress": [],
    }
    references: dict[str, list[torch.Tensor]] = {
        "energy": [],
        "forces": [],
        "stress": [],
    }
    meta: dict[str, list[torch.Tensor]] = {"n_atoms": [], "atomic_numbers": []}
    baseline_chunks: list[dict[str, torch.Tensor]] = []
    for member_position, member_id in enumerate(member_ids):
        paths = listed.get(member_id)
        expected_paths = [
            f"{member_id}/chunk_{index:03d}.pt" for index in range(expected.chunk_count)
        ]
        if paths != expected_paths:
            raise HardFailure(f"legacy chunk listing is invalid: {member_id}")
        member_dir = _directory(root, prediction_root / member_id, "member chunks")
        actual_names = sorted(path.name for path in member_dir.glob("chunk_*.pt"))
        if actual_names != [
            f"chunk_{index:03d}.pt" for index in range(expected.chunk_count)
        ]:
            raise HardFailure(f"legacy chunk files are incomplete: {member_id}")
        for chunk_index, (start, count) in enumerate(chunk_specs):
            path = _inside(
                root,
                prediction_root / expected_paths[chunk_index],
                "prediction chunk",
            )
            payload = _torch(path, snapshot)
            atom_counts = n_atoms_summary[start : start + count]
            A_chunk = sum(atom_counts)
            raw_structure_ids = payload.get("structure_ids")
            if not isinstance(raw_structure_ids, Sequence) or isinstance(
                raw_structure_ids, (str, bytes)
            ):
                raise HardFailure("legacy chunk structure IDs are invalid")
            if (
                payload.get("member_id") != member_id
                or payload.get("split") != "test"
                or payload.get("chunk_index") != chunk_index
                or payload.get("structure_start") != start
                or tuple(str(value) for value in raw_structure_ids)
                != structure_ids[start : start + count]
            ):
                raise HardFailure("legacy chunk identity is invalid")
            checked = {
                "n_atoms": _tensor(payload, "n_atoms", (count,), torch.int64),
                "atomic_numbers": _tensor(
                    payload, "atomic_numbers", (A_chunk,), torch.int64
                ),
                "energy_reference": _tensor(payload, "E_ref", (count,), torch.float32),
                "forces_reference": _tensor(
                    payload, "F_ref", (A_chunk, 3), torch.float32
                ),
                "stress_reference": _tensor(
                    payload, "S_ref", (count, 3, 3), torch.float32
                ),
            }
            if (
                tuple(int(value) for value in checked["n_atoms"].tolist())
                != atom_counts
            ):
                raise HardFailure("legacy chunk n_atoms differs from summary")
            offsets = _tensor(payload, "atom_offsets", (count + 1,), torch.int64)
            expected_offsets = torch.cat(
                (torch.zeros(1, dtype=torch.int64), checked["n_atoms"].cumsum(0))
            )
            mapping = _tensor(payload, "atom_to_structure", (A_chunk,), torch.int64)
            expected_mapping = torch.repeat_interleave(
                torch.arange(count, dtype=torch.int64), checked["n_atoms"]
            )
            if not torch.equal(offsets, expected_offsets) or not torch.equal(
                mapping, expected_mapping
            ):
                raise HardFailure("legacy chunk atom mapping is invalid")
            if member_position == 0:
                baseline_chunks.append(checked)
                for key in references:
                    references[key].append(checked[f"{key}_reference"])
                meta["n_atoms"].append(checked["n_atoms"])
                meta["atomic_numbers"].append(checked["atomic_numbers"])
            else:
                baseline = baseline_chunks[chunk_index]
                if any(not torch.equal(checked[key], baseline[key]) for key in checked):
                    raise HardFailure(
                        "legacy cross-member reference or mapping differs"
                    )
            predictions["energy"].append(
                _tensor(payload, "E_member", (count,), torch.float32)
            )
            predictions["forces"].append(
                _tensor(payload, "F_member", (A_chunk, 3), torch.float32)
            )
            predictions["stress"].append(
                _tensor(payload, "S_member", (count, 3, 3), torch.float32)
            )

    n_atoms = torch.cat(meta["n_atoms"])
    A = int(n_atoms.sum().item())
    member_chunks = expected.chunk_count
    prediction = {
        "energy_prediction": torch.stack(
            [
                torch.cat(
                    predictions["energy"][i * member_chunks : (i + 1) * member_chunks]
                )
                for i in range(len(member_ids))
            ]
        ),
        "forces_prediction": torch.stack(
            [
                torch.cat(
                    predictions["forces"][i * member_chunks : (i + 1) * member_chunks]
                )
                for i in range(len(member_ids))
            ]
        ),
        "stress_prediction": torch.stack(
            [
                torch.cat(
                    predictions["stress"][i * member_chunks : (i + 1) * member_chunks]
                )
                for i in range(len(member_ids))
            ]
        ),
        "energy_reference": torch.cat(references["energy"]),
        "forces_reference": torch.cat(references["forces"]),
        "stress_reference": torch.cat(references["stress"]),
        "n_atoms": n_atoms,
        "structure_offsets": torch.cat(
            (torch.zeros(1, dtype=torch.int64), n_atoms.cumsum(0))
        ),
        "member_ids": member_ids,
        "structure_ids": structure_ids,
        "atomic_numbers": torch.cat(meta["atomic_numbers"]),
        "structure_mapping": torch.repeat_interleave(
            torch.arange(S, dtype=torch.int64), n_atoms
        ),
        "target_names": {
            "energy": "energy",
            "forces": "non_conservative_forces",
            "stress": "non_conservative_stress",
        },
        "units": {
            "energy": "eV",
            "forces": "eV/angstrom",
            "stress": "eV/angstrom^3",
        },
        "statistics": {"K": len(member_ids), "S": S, "A": A},
    }
    return canonical_prediction(prediction)


def _risk_coverage(
    source: Path, expected: LegacyExpectations, snapshot: _SnapshotBuilder
) -> dict[str, tuple[Mapping[str, object], ...]]:
    result: dict[str, tuple[Mapping[str, object], ...]] = {}
    for role, raw_path in (expected.risk_coverage_paths or {}).items():
        if not isinstance(role, str) or not role:
            raise HardFailure("legacy risk coverage role is invalid")
        path = _inside(source, raw_path, "legacy risk coverage")
        snapshot.record(path)
        try:
            with path.open("r", encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                if reader.fieldnames != ["coverage", "risk"]:
                    raise HardFailure("legacy risk coverage header is invalid")
                rows = []
                for row in reader:
                    coverage = float(row["coverage"])
                    risk = float(row["risk"])
                    if not (math.isfinite(coverage) and math.isfinite(risk)):
                        raise HardFailure(
                            "legacy risk coverage contains non-finite values"
                        )
                    rows.append({"coverage": coverage, "risk": risk})
        except (OSError, ValueError, TypeError) as exc:
            raise HardFailure("legacy risk coverage is invalid") from exc
        if not rows:
            raise HardFailure("legacy risk coverage is empty")
        result[role] = tuple(rows)
    return result


def verify_source_unchanged(snapshot: SourceSnapshot) -> None:
    """Require every snapshotted critical source file to remain byte-identical."""

    for relative, expected_sha in snapshot.files:
        path = _inside(snapshot.root, relative, "snapshotted source")
        if sha256_file(path) != expected_sha:
            raise HardFailure(f"source changed after inspection: {relative}")


def read_legacy_run(root: Path, expected: LegacyExpectations) -> LegacyRun:
    """Validate and stream one successful legacy run into source-free values."""

    source = _source_root(root)
    member_ids = _expectation_schema(expected)
    snapshot = _SnapshotBuilder(source)
    manifest_path = _inside(source, expected.manifest_path, "legacy manifest")
    manifest = _json(manifest_path, snapshot)
    members = _members(source, manifest, expected, member_ids, snapshot)
    prediction = _prediction(source, expected, member_ids, snapshot)
    dataset = manifest.get("datasets")
    if (
        manifest.get("inference_only") is not True
        or not isinstance(dataset, Mapping)
        or not isinstance(dataset.get("test"), Mapping)
        or dataset["test"].get("sha256") != expected.test_data_sha256
    ):
        raise HardFailure("legacy result manifest test identity is invalid")
    uncertainty = _torch(
        _inside(source, expected.uncertainty_path, "legacy uncertainty"), snapshot
    )
    metrics = _json(_inside(source, expected.metrics_path, "legacy metrics"), snapshot)
    risk_coverage = _risk_coverage(source, expected, snapshot)
    if uncertainty.get("member_ids") != list(member_ids):
        raise HardFailure("legacy uncertainty member identity is invalid")
    return LegacyRun(
        members=members,
        prediction=prediction,
        legacy_uncertainty=uncertainty,
        legacy_metrics=metrics,
        legacy_risk_coverage=risk_coverage,
        source_snapshot=snapshot.finish(),
    )
