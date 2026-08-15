"""Publish a small, verifiable view of a complete remote evaluation."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from .artifacts import (
    atomic_json_dump,
    load_verified_manifest,
    sha256_file,
    stage_identity,
)


def export_evaluation_summary(evaluation_dir: Path, destination: Path) -> Path:
    """Copy summary/preview while binding the omitted full details by hash."""
    source = Path(evaluation_dir).resolve()
    output = Path(destination).resolve()
    source_manifest = load_verified_manifest(source / "manifest.json", verify_npz=True)
    declared = source_manifest.get("files")
    if not isinstance(declared, dict):
        raise ValueError("evaluation manifest files must be an object")
    required = ("summary.json", "preview.json", "details.npz")
    if any(not isinstance(declared.get(name), str) for name in required):
        raise ValueError(
            "evaluation manifest must declare summary, preview, and details"
        )
    if output.exists():
        raise FileExistsError(f"destination already exists: {output}")

    identity = stage_identity(
        "evaluation-summary",
        {
            "evaluation_identity": source_manifest["identity"],
            "details_sha256": declared["details.npz"],
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir()
    try:
        for name in ("summary.json", "preview.json"):
            shutil.copy2(source / name, staging / name)
        files = {
            name: sha256_file(staging / name)
            for name in ("summary.json", "preview.json")
        }
        atomic_json_dump(
            staging / "manifest.json",
            {
                **identity,
                "status": "complete",
                "evaluation_identity": source_manifest["identity"],
                "details_sha256": declared["details.npz"],
                "files": files,
            },
        )
        load_verified_manifest(staging / "manifest.json")
        if output.exists():
            raise FileExistsError(f"destination already exists: {output}")
        staging.replace(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output
