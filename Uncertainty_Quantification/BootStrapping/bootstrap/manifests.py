"""Creation and recursive validation of immutable run manifests."""

from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any, Iterable, Mapping

from .artifacts import atomic_write_json, sha256_file
from .errors import HardFailure


def _inside(root: Path, path: Path) -> Path:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise HardFailure(f"artifact escapes run root: {path}") from error
    if relative == Path(".") or ".." in relative.parts:
        raise HardFailure(f"artifact escapes run root: {path}")
    return relative


def _regular_non_symlink(path: Path) -> None:
    if path.is_symlink():
        raise HardFailure(f"manifest artifact is a symlink: {path}")
    try:
        mode = path.stat().st_mode
    except OSError as error:
        raise HardFailure(f"manifest artifact is missing: {path}") from error
    if not stat.S_ISREG(mode):
        raise HardFailure(f"manifest artifact is not a regular file: {path}")


def build_run_manifest(
    root: str | Path,
    *,
    schema: str,
    artifacts: Iterable[str | Path],
    metadata: Mapping[str, Any] | None = None,
    filename: str = "run_manifest.json",
) -> Path:
    """Audit files below ``root`` and publish their canonical manifest."""

    root_path = Path(root).expanduser().absolute()
    entries: list[dict[str, Any]] = []
    for artifact in artifacts:
        artifact_path = Path(artifact).expanduser().absolute()
        relative = _inside(root_path, artifact_path)
        _regular_non_symlink(artifact_path)
        entries.append(
            {
                "path": relative.as_posix(),
                "sha256": sha256_file(artifact_path),
                "size_bytes": artifact_path.stat().st_size,
            }
        )
    document = {
        "schema": schema,
        "metadata": dict(metadata or {}),
        "artifacts": sorted(entries, key=lambda item: item["path"]),
    }
    return atomic_write_json(root_path / filename, document)


def validate_manifest(path: str | Path, *, expected_schema: str) -> dict[str, object]:
    """Reject schema drift, escapes, missing artifacts and SHA drift."""

    manifest_path = Path(path).expanduser().absolute()
    _regular_non_symlink(manifest_path)
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HardFailure(
            f"could not load manifest {manifest_path}: {error}"
        ) from error
    if not isinstance(document, dict):
        raise HardFailure("manifest root must be an object")
    if set(document) != {"schema", "metadata", "artifacts"}:
        raise HardFailure("manifest has unknown or missing root keys")
    if document["schema"] != expected_schema:
        raise HardFailure(
            f"manifest schema must be {expected_schema}, got {document['schema']}"
        )
    if not isinstance(document["metadata"], dict):
        raise HardFailure("manifest metadata must be an object")
    if not isinstance(document["artifacts"], list):
        raise HardFailure("manifest artifacts must be a list")

    root = manifest_path.parent
    for entry in document["artifacts"]:
        if not isinstance(entry, dict) or set(entry) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise HardFailure("manifest artifact entry is invalid")
        relative_text = entry["path"]
        if not isinstance(relative_text, str):
            raise HardFailure("manifest artifact path must be a string")
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
            raise HardFailure(f"manifest artifact escapes run root: {relative_text}")
        artifact = root / relative
        _inside(root, artifact)
        _regular_non_symlink(artifact)
        if artifact.stat().st_size != entry["size_bytes"]:
            raise HardFailure(f"manifest artifact size drift: {relative_text}")
        if sha256_file(artifact) != entry["sha256"]:
            raise HardFailure(f"manifest artifact SHA-256 drift: {relative_text}")
    return document
