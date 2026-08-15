"""Strict validation and transactional materialization of reusable LLPR stages."""

from __future__ import annotations

import errno
import json
import os
import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

from .artifacts import (
    FORMULA_VERSION,
    SCHEMA_VERSION,
    canonical_json,
    declared_artifact_paths,
    load_verified_manifest,
)


StageName = Literal["curvature", "calibration"]
REQUIRED_STAGE_FILES: dict[StageName, frozenset[str]] = {
    "curvature": frozenset({"curvature.npz", "diagnostics.json"}),
    "calibration": frozenset({"candidates.json", "summary.json"}),
}
LINK_FALLBACK_ERRNOS = frozenset(
    {errno.EXDEV, errno.EPERM, errno.EACCES, errno.EOPNOTSUPP}
)


class _TargetLayout(Protocol):
    dimension: int


class CurvatureLayout(Protocol):
    energy: _TargetLayout
    force: _TargetLayout
    layout_hash: str


def _manifest_files(manifest: Mapping[str, object]) -> dict[str, str]:
    value = manifest.get("files")
    if not isinstance(value, dict) or not all(
        isinstance(name, str) and isinstance(digest, str)
        for name, digest in value.items()
    ):
        raise ValueError("manifest files must map strings to strings")
    return {str(name): str(digest) for name, digest in value.items()}


def _validate_stage_manifest(
    source: Path,
    *,
    stage: StageName,
    identity: str,
    expected_payload: Mapping[str, object],
    expected_curvature_identity: str | None,
) -> dict[str, object]:
    manifest = load_verified_manifest(
        source / "manifest.json",
        {
            "stage": stage,
            "identity": identity,
            "schema_version": SCHEMA_VERSION,
            "formula_version": FORMULA_VERSION,
        },
        verify_npz=True,
    )
    files = _manifest_files(manifest)
    missing = sorted(REQUIRED_STAGE_FILES[stage].difference(files))
    if missing:
        raise ValueError(f"reused {stage} manifest is missing files: {missing}")
    if expected_curvature_identity is not None:
        actual_curvature = manifest.get("curvature_identity")
        if actual_curvature != expected_curvature_identity:
            raise ValueError(
                "reused calibration curvature identity mismatch: "
                f"{actual_curvature!r} != {expected_curvature_identity!r}"
            )
    payload = manifest.get("payload")
    if not isinstance(payload, dict):
        raise ValueError(f"reused {stage} manifest payload must be an object")
    for name, expected in expected_payload.items():
        actual = payload.get(name)
        if canonical_json(actual) != canonical_json(expected):
            raise ValueError(
                f"reused {stage} payload field {name!r} mismatch: "
                f"{actual!r} != {expected!r}"
            )
    return manifest


def _copy_or_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError as error:
        if error.errno not in LINK_FALLBACK_ERRNOS:
            raise
        shutil.copy2(source, destination)


def _is_nested(left: Path, right: Path) -> bool:
    """Return whether either resolved directory contains the other."""
    return left == right or left in right.parents or right in left.parents


def materialize_reused_stage(
    source: Path,
    destination_root: Path,
    *,
    stage: StageName,
    identity: str,
    expected_payload: Mapping[str, object],
    expected_curvature_identity: str | None = None,
) -> Path:
    """Verify one stage and atomically expose it below a new experiment root."""
    source_input = Path(source)
    if source_input.is_symlink():
        raise ValueError(f"reuse source must be an ordinary directory: {source_input}")
    source = source_input.resolve()
    if not source.is_dir():
        raise ValueError(f"reuse source must be an ordinary directory: {source}")
    destination_root = Path(destination_root).resolve()
    target = destination_root / identity
    if _is_nested(source, target):
        raise ValueError("reuse source and target must be separate directories")
    source_manifest = _validate_stage_manifest(
        source,
        stage=stage,
        identity=identity,
        expected_payload=expected_payload,
        expected_curvature_identity=expected_curvature_identity,
    )
    source_files = _manifest_files(source_manifest)

    if target.exists():
        try:
            target_manifest = _validate_stage_manifest(
                target,
                stage=stage,
                identity=identity,
                expected_payload=expected_payload,
                expected_curvature_identity=expected_curvature_identity,
            )
            if canonical_json(_manifest_files(target_manifest)) != canonical_json(
                source_files
            ):
                raise ValueError("declared file hashes differ from reuse source")
        except (OSError, ValueError) as error:
            raise ValueError(
                f"existing reuse target is incompatible: {target}"
            ) from error
        return target

    destination_root.mkdir(parents=True, exist_ok=True)
    staging = destination_root / f".{identity}.{uuid.uuid4().hex}.staging"
    staging.mkdir()
    try:
        for relative, source_path in declared_artifact_paths(
            source, source_manifest
        ).items():
            _copy_or_link(source_path, staging / relative)
        _copy_or_link(source / "manifest.json", staging / "manifest.json")
        staged_manifest = _validate_stage_manifest(
            staging,
            stage=stage,
            identity=identity,
            expected_payload=expected_payload,
            expected_curvature_identity=expected_curvature_identity,
        )
        if canonical_json(_manifest_files(staged_manifest)) != canonical_json(
            source_files
        ):
            raise ValueError("staged reuse file hashes differ from source")
        staging.replace(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def validate_curvature_layout(
    stage_dir: Path,
    manifest: Mapping[str, object],
    layout: CurvatureLayout,
) -> None:
    """Bind a verified curvature artifact to the checkpoint's readout layout."""
    stage_dir = Path(stage_dir)
    verified = load_verified_manifest(
        stage_dir / "manifest.json",
        {
            "stage": "curvature",
            "identity": manifest.get("identity"),
            "schema_version": SCHEMA_VERSION,
            "formula_version": FORMULA_VERSION,
        },
        verify_npz=True,
    )
    if verified.get("layout_hash") != layout.layout_hash:
        raise ValueError(
            "curvature layout hash mismatch: "
            f"{verified.get('layout_hash')!r} != {layout.layout_hash!r}"
        )
    diagnostics = json.loads(
        (stage_dir / "diagnostics.json").read_text(encoding="utf-8")
    )
    if not isinstance(diagnostics, dict):
        raise ValueError("curvature diagnostics must contain an object")
    with np.load(stage_dir / "curvature.npz", allow_pickle=False) as archive:
        matrices = {
            "energy": np.asarray(archive["energy"]),
            "force": np.asarray(archive["force"]),
        }
    expected_dimensions = {
        "energy": layout.energy.dimension,
        "force": layout.force.dimension,
    }
    for target, expected_dimension in expected_dimensions.items():
        matrix = matrices[target]
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"{target} curvature matrix must be square")
        if matrix.shape[0] != expected_dimension:
            raise ValueError(
                f"{target} matrix dimension mismatch: "
                f"{matrix.shape[0]} != {expected_dimension}"
            )
        diagnostic_dimension = diagnostics.get(f"{target}_dimension")
        if diagnostic_dimension != matrix.shape[0]:
            raise ValueError(
                f"{target} diagnostics dimension mismatch: "
                f"{diagnostic_dimension!r} != {matrix.shape[0]}"
            )
    if diagnostics.get("total_dimension") != sum(expected_dimensions.values()):
        raise ValueError("curvature diagnostics total dimension mismatch")
