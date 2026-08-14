"""Validated compare-and-swap finalization of a bootstrap run."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

from .errors import HardFailure
from .validation import validate_uq_publication


def _read_manifest(path: Path) -> tuple[bytes, dict[str, Any]]:
    try:
        payload = path.read_bytes()
        document = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise HardFailure(f"could not load run manifest {path}: {error}") from error
    if not isinstance(document, dict):
        raise HardFailure(f"run manifest must contain an object: {path}")
    return payload, document


def finalize_run(
    run_root: str | Path,
    *,
    splits: Iterable[str],
    modes: Iterable[str],
    member_count: int,
) -> dict[str, Any]:
    """Validate all requested UQ artifacts, then mark their run complete."""

    root = Path(run_root).expanduser().resolve()
    combinations = tuple(sorted({(split, mode) for split in splits for mode in modes}))
    if not combinations:
        raise HardFailure("run finalization requires at least one split/mode")
    manifest_refs: list[str] = []
    for split, mode in combinations:
        validate_uq_publication(
            root, split=split, mode=mode, member_count=member_count
        )
        manifest_refs.append(f"uncertainty/{split}/{mode}/manifest.json")

    path = root / "run_manifest.json"
    original, manifest = _read_manifest(path)
    if manifest.get("schema") != "upet.bootstrap.run/v1":
        raise HardFailure(f"unexpected run manifest schema: {path}")
    stages = manifest.get("stages")
    if not isinstance(stages, dict):
        raise HardFailure(f"run manifest has no stages object: {path}")
    state = stages.get("uncertainty")
    if state == "complete":
        if manifest.get("uncertainty_manifests") != manifest_refs:
            raise HardFailure(f"completed run has different UQ publications: {path}")
        return manifest
    if state != "pending":
        raise HardFailure(f"run uncertainty stage is not pending: {path}")

    updated = dict(manifest)
    updated_stages = dict(stages)
    updated_stages["uncertainty"] = "complete"
    updated["stages"] = updated_stages
    updated["uncertainty_manifests"] = manifest_refs
    payload = (
        json.dumps(updated, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        current, _ = _read_manifest(path)
        if current != original:
            raise HardFailure(f"run manifest changed during finalization: {path}")
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except HardFailure:
        raise
    except OSError as error:
        raise HardFailure(f"could not finalize run manifest {path}: {error}") from error
    finally:
        temporary.unlink(missing_ok=True)
    return updated
