"""Hashing, transactional writes, and manifest validation."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


def sha256_file(path: Path) -> str:
    """Calculate a file's SHA-256 digest without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _temporary_path(path: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    return Path(name)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Replace a JSON target only after writing and validating a complete object."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(target)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        verified = json.loads(temporary.read_text(encoding="utf-8"))
        if not isinstance(verified, Mapping):
            raise ValueError("JSON artifact must contain a mapping")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    """Replace a torch target only after it can be loaded as a mapping."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(target)
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        verified = torch.load(
            temporary,
            weights_only=False,
            map_location="cpu",
        )
        if not isinstance(verified, Mapping):
            raise ValueError("torch artifact must contain a mapping")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def require_complete_manifest(
    path: Path,
    expected_identity: str,
) -> dict[str, Any]:
    """Load a complete manifest whose identity exactly matches the expectation."""
    manifest_path = Path(path)
    try:
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid manifest {manifest_path}: {error}") from error
    if not isinstance(loaded, dict):
        raise ValueError(f"manifest {manifest_path} must contain a mapping")
    if loaded.get("status") != "complete":
        raise ValueError(f"manifest status must be complete: {manifest_path}")
    if loaded.get("identity") != expected_identity:
        raise ValueError(
            "manifest identity mismatch: "
            f"{loaded.get('identity')!r} != {expected_identity!r}"
        )
    return loaded
