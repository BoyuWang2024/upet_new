"""Read-only validation of published bootstrap uncertainty artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .artifacts import _absolute_lexical, _reject_symlink_components
from .errors import HardFailure
from .identifiers import validate_artifact_key
from .uq_publication import _manifest_document, compute_uncertainty_results


_PUBLIC_UNITS = {
    "energy": "eV",
    "forces": "eV/Angstrom",
    "stress": "eV/Angstrom^3",
}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise HardFailure(f"could not load JSON artifact {path}: {error}") from error
    if not isinstance(document, dict):
        raise HardFailure(f"JSON artifact must contain an object: {path}")
    return document


def _validate_request(split: object, mode: object, member_count: object) -> str:
    key = validate_artifact_key(split, "uncertainty dataset key")
    if mode not in {"raw", "ema"}:
        raise HardFailure("uncertainty mode must be raw or ema")
    if (
        isinstance(member_count, bool)
        or not isinstance(member_count, int)
        or member_count < 2
    ):
        raise HardFailure("uncertainty member_count must be at least 2")
    return key


def _exact_publication_files(root: Path) -> tuple[Path, Path]:
    if not root.is_dir() or root.is_symlink():
        raise HardFailure(f"UQ publication is not a directory: {root}")
    try:
        children = {path.name: path for path in root.iterdir()}
    except OSError as error:
        raise HardFailure(
            f"could not inspect UQ publication {root}: {error}"
        ) from error
    if set(children) != {"results.npz", "manifest.json"}:
        raise HardFailure(
            f"UQ publication must contain the exact canonical file set: {root}"
        )
    results_path = children["results.npz"]
    manifest_path = children["manifest.json"]
    if any(
        not path.is_file() or path.is_symlink()
        for path in (results_path, manifest_path)
    ):
        raise HardFailure(f"UQ publication files must be regular files: {root}")
    return results_path, manifest_path


def validate_uq_publication(
    run_root: str | Path,
    *,
    split: str,
    mode: str,
    member_count: int,
    units: Mapping[str, str] = _PUBLIC_UNITS,
    publication_root: str | Path | None = None,
) -> dict[str, Any]:
    """Recompute one generic UQ publication and require complete exact equality."""

    key = _validate_request(split, mode, member_count)
    root_path = _absolute_lexical(run_root)
    _reject_symlink_components(root_path)
    root = root_path.resolve()
    expected = compute_uncertainty_results(
        root / "predictions" / key, mode=mode, member_count=member_count
    )
    publication_path = (
        _absolute_lexical(publication_root)
        if publication_root is not None
        else root_path / "uncertainty" / key / mode
    )
    _reject_symlink_components(publication_path)
    published = publication_path.resolve()
    results_path, manifest_path = _exact_publication_files(published)
    try:
        with np.load(results_path, allow_pickle=False) as archive:
            if set(archive.files) != set(expected):
                raise HardFailure(
                    f"UQ result keys do not match the canonical schema: {results_path}"
                )
            actual = {
                name: np.array(archive[name], copy=True) for name in archive.files
            }
    except HardFailure:
        raise
    except (OSError, ValueError) as error:
        raise HardFailure(
            f"could not load UQ results {results_path}: {error}"
        ) from error
    for name, expected_array in expected.items():
        actual_array = actual[name]
        if actual_array.dtype != np.dtype(np.float64):
            raise HardFailure(f"UQ result must use float64: {results_path}:{name}")
        if not bool(np.isfinite(actual_array).all()):
            raise HardFailure(
                f"UQ result contains non-finite values: {results_path}:{name}"
            )
        if not np.array_equal(actual_array, expected_array):
            raise HardFailure(
                f"UQ result does not match recomputation: {results_path}:{name}"
            )
    manifest = _load_json(manifest_path)
    expected_manifest = _manifest_document(
        results_path,
        expected,
        dataset_key=key,
        mode=mode,
        member_count=member_count,
        units=units,
    )
    if manifest != expected_manifest:
        raise HardFailure(
            f"UQ manifest does not exactly match canonical v1: {manifest_path}"
        )
    return manifest
