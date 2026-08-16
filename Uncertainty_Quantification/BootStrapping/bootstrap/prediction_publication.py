"""Validation for complete native prediction publications."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .artifacts import _reject_symlink_components, sha256_file
from .errors import HardFailure
from .identifiers import validate_artifact_key
from .prediction import (
    TargetArrays,
    load_prediction_arrays,
    load_target_arrays,
    validate_predictions,
)
from .prediction import (
    reference_targets as target_fields,
)


_UNITS = {"energy": "eV", "forces": "eV/Angstrom", "stress": "eV/Angstrom^3"}
_MEMBER_KEYS = {"member_index", "mode", "path", "sha256", "shapes", "dtypes"}


@dataclass(frozen=True)
class PredictionPublicationAudit:
    """A verified immutable prediction publication."""

    manifest_path: Path
    dataset_key: str
    mode: str
    member_count: int


def _mapping(value: object, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HardFailure(f"{location} must be a mapping")
    return value


def _publication_root(value: str | Path) -> Path:
    root = Path(value).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    _reject_symlink_components(root)
    return root


def _load_manifest(path: Path) -> Mapping[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HardFailure(
            f"could not load prediction manifest {path}: {error}"
        ) from error
    return _mapping(document, "prediction manifest")


def _expected_metadata(
    values: object,
) -> tuple[dict[str, list[int]], dict[str, str]]:
    return (
        {
            name: list(getattr(values, name).shape)
            for name in ("energy", "forces", "stress")
        },
        {
            name: str(getattr(values, name).dtype)
            for name in ("energy", "forces", "stress")
        },
    )


def _validate_targets(
    root: Path,
    document: Mapping[str, Any],
    expected_reference_targets: tuple[str, ...],
    expected_structure_limit: int | None,
) -> TargetArrays:
    targets = _mapping(document.get("targets"), "prediction manifest.targets")
    if set(targets) != {"structure_limit", "path", "sha256"}:
        raise HardFailure("prediction manifest.targets keys do not match schema")
    actual_limit = targets["structure_limit"]
    if expected_structure_limit is None:
        if actual_limit is not None:
            raise HardFailure("prediction manifest structure_limit differs")
    elif (
        isinstance(actual_limit, bool)
        or not isinstance(actual_limit, int)
        or actual_limit != expected_structure_limit
    ):
        raise HardFailure("prediction manifest structure_limit differs")
    if targets["path"] != "targets.npz":
        raise HardFailure("prediction manifest targets path does not match schema")
    target_path = root / "targets.npz"
    if targets["sha256"] != sha256_file(target_path):
        raise HardFailure("prediction manifest targets SHA-256 differs")
    loaded = load_target_arrays(target_path)
    if target_fields(loaded) != expected_reference_targets:
        raise HardFailure("prediction manifest reference targets differ")
    return loaded


def _validate_members(
    root: Path,
    document: Mapping[str, Any],
    *,
    modes: tuple[str, ...],
    member_count: int,
    targets: TargetArrays,
) -> None:
    records = document.get("members")
    if not isinstance(records, list) or len(records) != member_count * len(modes):
        raise HardFailure("prediction manifest member records do not match count")
    declared_paths: set[str] = set()
    for position, record in enumerate(records):
        item = _mapping(record, "prediction manifest member")
        if set(item) != _MEMBER_KEYS:
            raise HardFailure(
                "prediction manifest member record keys do not match schema"
            )
        index = item["member_index"]
        expected_index = position // len(modes)
        expected_mode = modes[position % len(modes)]
        if isinstance(index, bool) or not isinstance(index, int):
            raise HardFailure("prediction manifest member_index must be an integer")
        if index != expected_index:
            raise HardFailure("prediction manifest member record order differs")
        expected_path = f"members/member_{index:03d}/{expected_mode}.npz"
        if (
            index < 0
            or item["mode"] != expected_mode
            or item["path"] != expected_path
            or item["path"] in declared_paths
        ):
            raise HardFailure("prediction manifest member records do not match schema")
        member_path = root / expected_path
        if item["sha256"] != sha256_file(member_path):
            raise HardFailure(f"prediction member SHA-256 differs: {index}")
        values = load_prediction_arrays(member_path)
        validate_predictions(values, targets)
        shapes, dtypes = _expected_metadata(values)
        if item["shapes"] != shapes or item["dtypes"] != dtypes:
            raise HardFailure(f"prediction member metadata differs: {index}")
        declared_paths.add(expected_path)
    actual_paths = {
        str(path.relative_to(root)).replace("\\", "/")
        for path in (root / "members").rglob("*.npz")
    }
    if actual_paths != declared_paths:
        raise HardFailure("prediction member files do not match manifest")


def _validate_v1_header(
    document: Mapping[str, Any], dataset_key: str, member_count: int
) -> None:
    required = {
        "schema",
        "split",
        "units",
        "member_count",
        "targets",
        "members",
    }
    if (
        set(document) != required
        or document["split"] != dataset_key
        or document["units"] != _UNITS
        or isinstance(document["member_count"], bool)
        or not isinstance(document["member_count"], int)
        or document["member_count"] != member_count
    ):
        raise HardFailure("prediction manifest metadata does not match request")


def validate_prediction_publication(
    split_root: str | Path,
    *,
    dataset_key: str,
    mode: str,
    member_count: int,
    reference_targets: tuple[str, ...],
    dataset_label: str | None = None,
    structure_limit: int | None = None,
) -> PredictionPublicationAudit:
    """Audit a v1 or v2 publication before it can be reused."""

    key = validate_artifact_key(dataset_key, "prediction dataset key")
    if mode not in {"raw", "ema"}:
        raise HardFailure("prediction mode must be raw or ema")
    if isinstance(member_count, bool) or member_count < 1:
        raise HardFailure("prediction member_count must be positive")
    if reference_targets not in (("energy", "forces"), ("energy", "forces", "stress")):
        raise HardFailure("prediction reference targets are not supported")
    root = _publication_root(split_root)
    manifest_path = root / "manifest.json"
    document = _load_manifest(manifest_path)
    schema = document.get("schema")
    if schema == "upet.bootstrap.predictions/v1":
        _validate_v1_header(document, key, member_count)
        expected_schema_targets = ("energy", "forces", "stress")
    elif schema == "upet.bootstrap.predictions/v2":
        required = {
            "schema",
            "split",
            "dataset_key",
            "dataset_label",
            "reference_targets",
            "units",
            "member_count",
            "targets",
            "members",
        }
        manifest_dataset_label = document.get("dataset_label")
        if not isinstance(manifest_dataset_label, str) or (
            dataset_label is not None and manifest_dataset_label != dataset_label
        ):
            raise HardFailure("prediction manifest dataset label differs")
        if (
            set(document) != required
            or document["split"] != key
            or document["dataset_key"] != key
            or document["reference_targets"] != list(reference_targets)
        ):
            raise HardFailure("prediction manifest dataset key does not match")
        expected_schema_targets = ("energy", "forces")
    else:
        raise HardFailure("prediction manifest schema is not supported")
    if schema != "upet.bootstrap.predictions/v1" and (
        document["units"] != _UNITS
        or isinstance(document["member_count"], bool)
        or not isinstance(document["member_count"], int)
        or document["member_count"] != member_count
    ):
        raise HardFailure("prediction manifest metadata does not match request")
    targets = _validate_targets(root, document, reference_targets, structure_limit)
    if target_fields(targets) != expected_schema_targets:
        raise HardFailure("prediction manifest schema target layout differs")
    _validate_members(
        root, document, modes=(mode,), member_count=member_count, targets=targets
    )
    return PredictionPublicationAudit(manifest_path, key, mode, member_count)


def validate_legacy_prediction_publication(
    split_root: str | Path,
    *,
    dataset_key: str,
    modes: tuple[str, ...],
    member_count: int,
    structure_limit: int | None,
) -> PredictionPublicationAudit:
    """Audit an ordered legacy raw-plus-EMA v1 publication."""

    key = validate_artifact_key(dataset_key, "prediction dataset key")
    if len(modes) != 2 or set(modes) != {"raw", "ema"}:
        raise HardFailure(
            "legacy prediction modes must contain raw and ema exactly once"
        )
    if isinstance(member_count, bool) or member_count < 1:
        raise HardFailure("prediction member_count must be positive")
    root = _publication_root(split_root)
    manifest_path = root / "manifest.json"
    document = _load_manifest(manifest_path)
    if document.get("schema") != "upet.bootstrap.predictions/v1":
        raise HardFailure("legacy prediction manifest schema is not supported")
    _validate_v1_header(document, key, member_count)
    targets = _validate_targets(
        root,
        document,
        ("energy", "forces", "stress"),
        structure_limit,
    )
    if target_fields(targets) != ("energy", "forces", "stress"):
        raise HardFailure("prediction manifest schema target layout differs")
    _validate_members(
        root, document, modes=modes, member_count=member_count, targets=targets
    )
    return PredictionPublicationAudit(manifest_path, key, "raw+ema", member_count)
