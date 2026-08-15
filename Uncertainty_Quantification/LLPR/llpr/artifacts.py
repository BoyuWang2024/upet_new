"""Canonical identities, transactional writers, and artifact verification."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np


SCHEMA_VERSION = "upet-llpr-artifact-v1"
FORMULA_VERSION = "upet-llpr-huber-readout-v1"


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__} as canonical JSON")


def canonical_json(value: object) -> str:
    """Return the compact, sorted JSON representation used for identities."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_default,
    )


def stable_id(value: object, length: int = 16) -> str:
    """Hash a JSON-compatible value to a stable hexadecimal identifier."""
    if length <= 0 or length > 64:
        raise ValueError("length must be between 1 and 64")
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return digest[:length]


def sha256_file(path: Path) -> str:
    """Calculate SHA-256 without loading the whole file into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_identity(stage: str, payload: Mapping[str, object]) -> dict[str, object]:
    """Bind one stage to the common schema/formula and relevant inputs."""
    identity_payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "formula_version": FORMULA_VERSION,
        "stage": stage,
        "payload": dict(payload),
    }
    return {**identity_payload, "identity": stable_id(identity_payload)}


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")


def atomic_json_dump(path: Path, value: Mapping[str, object]) -> None:
    """Write JSON by replacing the destination only after a complete write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        temporary.write_text(
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                allow_nan=False,
                default=_json_default,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_npz_save(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Atomically write compressed NumPy arrays with the exact target name."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_complete_manifest(
    path: Path,
    expected_identity: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Load a complete manifest and optionally match identity fields."""
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"manifest {path} must contain an object")
    if manifest.get("status") != "complete":
        raise ValueError(f"manifest status must be complete: {path}")
    if expected_identity is not None:
        mismatched = {
            key: (manifest.get(key), expected)
            for key, expected in expected_identity.items()
            if canonical_json(manifest.get(key)) != canonical_json(expected)
        }
        if mismatched:
            raise ValueError(f"manifest identity mismatch: {mismatched}")
    return manifest


@dataclass(frozen=True)
class RunPaths:
    """Pure path layout for one canonical experiment root."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))

    @property
    def curvature(self) -> Path:
        return self.root / "curvature"

    @property
    def calibration(self) -> Path:
        return self.root / "calibration"

    @property
    def evaluation(self) -> Path:
        return self.root / "evaluation"


def _resolve_declared_path(root: Path, relative_value: str) -> Path:
    relative = Path(relative_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"declared artifact path escapes manifest root: {relative}")
    resolved_root = Path(root).resolve()
    path = (resolved_root / relative).resolve()
    try:
        path.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(
            f"declared artifact path escapes manifest root: {relative}"
        ) from error
    return path


def declared_artifact_paths(
    root: Path, manifest: Mapping[str, object]
) -> dict[str, Path]:
    """Resolve the manifest's declared files without allowing root escapes."""
    files = manifest.get("files", {})
    if not isinstance(files, dict):
        raise ValueError("manifest files must be an object")
    resolved: dict[str, Path] = {}
    for relative, expected_sha in files.items():
        if not isinstance(relative, str) or not isinstance(expected_sha, str):
            raise ValueError("manifest file hashes must map strings to strings")
        resolved[relative] = _resolve_declared_path(root, relative)
    return resolved


def _verify_declared_files(
    root: Path,
    manifest: Mapping[str, object],
    *,
    verify_npz: bool = False,
) -> int:
    files = cast(dict[str, str], manifest.get("files", {}))
    paths = declared_artifact_paths(root, manifest)
    verified = 0
    for relative, path in paths.items():
        expected_sha = files[relative]
        assert isinstance(expected_sha, str)
        if not path.is_file():
            raise ValueError(f"declared artifact is missing: {path}")
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise ValueError(
                f"artifact SHA mismatch for {path}: {actual_sha} != {expected_sha}"
            )
        if verify_npz and path.suffix == ".npz":
            _verify_npz(path)
        verified += 1
    return verified


def load_verified_manifest(
    path: Path,
    expected_identity: Mapping[str, object] | None = None,
    *,
    verify_npz: bool = False,
) -> dict[str, object]:
    """Load a complete manifest and verify all declared artifacts."""
    path = Path(path)
    manifest = load_complete_manifest(path, expected_identity)
    _verify_declared_files(path.parent, manifest, verify_npz=verify_npz)
    return manifest


def publish_run_manifest(
    root: Path,
    *,
    curvature_manifest: Mapping[str, object],
    calibration_manifest: Mapping[str, object],
    evaluation_manifest: Mapping[str, object],
) -> Path:
    """Publish the neutral root manifest for one complete LLPR run."""
    manifests = {
        "curvature_identity": curvature_manifest,
        "calibration_identity": calibration_manifest,
        "evaluation_identity": evaluation_manifest,
    }
    identities: dict[str, str] = {}
    for name, manifest in manifests.items():
        value = manifest.get("identity")
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} manifest has no valid identity")
        identities[name] = value

    identity = stage_identity("run", identities)
    manifest = {
        **identity,
        "status": "complete",
        **identities,
        "files": {},
    }
    path = Path(root) / "manifest.json"
    if path.exists():
        existing = load_complete_manifest(path)
        if existing.get("identity") != identity["identity"]:
            raise ValueError(
                "run root already contains a different complete experiment"
            )
        return path
    atomic_json_dump(path, manifest)
    return path


def _verify_npz(path: Path) -> None:
    with np.load(path, allow_pickle=False) as archive:
        for name in archive.files:
            array = archive[name]
            if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
                raise ValueError(f"non-finite array {name!r} in {path}")


def verify_run(
    root: Path, level: Literal["metadata", "full"] = "metadata"
) -> dict[str, object]:
    """Verify complete manifests and their declared file hashes."""
    root = Path(root)
    manifests = sorted(root.rglob("manifest.json"))
    if not manifests:
        raise ValueError(f"no manifests found under {root}")
    verified_files = 0
    for path in manifests:
        manifest = load_complete_manifest(path)
        verified_files += _verify_declared_files(
            path.parent, manifest, verify_npz=level == "full"
        )
    if not math.isfinite(float(verified_files)):
        raise AssertionError("unreachable non-finite file count")
    return {
        "status": "complete",
        "level": level,
        "manifest_count": len(manifests),
        "verified_file_count": verified_files,
    }
