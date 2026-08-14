"""Read-only validation of published bootstrap uncertainty artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import sha256_file
from .errors import HardFailure
from .prediction import load_target_arrays
from .uncertainty import pairwise_gmd, scalar_rms_reductions, streaming_mean_std


_RESULT_KEYS = {
    "energy_mean",
    "energy_std",
    "energy_gmd",
    "forces_mean",
    "forces_std",
    "forces_gmd",
    "stress_mean",
    "stress_std",
    "stress_gmd",
    "energy_per_atom_std",
    "energy_per_atom_gmd",
    "force_vector_rms_std",
    "force_vector_rms_gmd",
    "stress_tensor_rms_std",
    "stress_tensor_rms_gmd",
}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise HardFailure(f"could not load JSON artifact {path}: {error}") from error
    if not isinstance(document, dict):
        raise HardFailure(f"JSON artifact must contain an object: {path}")
    return document


def validate_uq_publication(
    run_root: str | Path,
    *,
    split: str,
    mode: str,
    member_count: int,
) -> dict[str, Any]:
    """Recompute one UQ publication and require exact canonical equality."""

    if split not in {"val", "test"}:
        raise HardFailure("uncertainty split must be val or test")
    if mode not in {"raw", "ema"}:
        raise HardFailure("uncertainty mode must be raw or ema")
    if isinstance(member_count, bool) or member_count < 2:
        raise HardFailure("uncertainty member_count must be at least 2")

    root = Path(run_root).expanduser().resolve()
    targets = load_target_arrays(root / "predictions" / split / "targets.npz")
    members = tuple(
        root
        / "predictions"
        / split
        / "members"
        / f"member_{index:03d}"
        / f"{mode}.npz"
        for index in range(member_count)
    )
    if any(not path.is_file() for path in members):
        raise HardFailure(f"UQ validation requires every configured {mode} member")

    expected: dict[str, np.ndarray] = {}
    for field in ("energy", "forces", "stress"):
        statistics = streaming_mean_std(members, field)
        expected[f"{field}_mean"] = statistics.mean
        expected[f"{field}_std"] = statistics.std
        expected[f"{field}_gmd"] = pairwise_gmd(members, field)
    for measure in ("std", "gmd"):
        expected[f"energy_per_atom_{measure}"] = scalar_rms_reductions(
            expected[f"energy_{measure}"],
            "energy",
            num_atoms=targets.num_atoms,
            atom_offsets=targets.atom_offsets,
        )
        expected[f"force_vector_rms_{measure}"] = scalar_rms_reductions(
            expected[f"forces_{measure}"],
            "forces",
            num_atoms=targets.num_atoms,
            atom_offsets=targets.atom_offsets,
        )
        expected[f"stress_tensor_rms_{measure}"] = scalar_rms_reductions(
            expected[f"stress_{measure}"],
            "stress",
            num_atoms=targets.num_atoms,
            atom_offsets=targets.atom_offsets,
        )

    publication_root = root / "uncertainty" / split / mode
    results_path = publication_root / "results.npz"
    try:
        with np.load(results_path, allow_pickle=False) as archive:
            if set(archive.files) != _RESULT_KEYS:
                raise HardFailure(
                    f"UQ result keys do not match the canonical schema: {results_path}"
                )
            actual = {name: np.array(archive[name], copy=True) for name in archive.files}
    except HardFailure:
        raise
    except (OSError, ValueError) as error:
        raise HardFailure(f"could not load UQ results {results_path}: {error}") from error

    for name, expected_array in expected.items():
        actual_array = actual[name]
        if actual_array.dtype != np.dtype(np.float64):
            raise HardFailure(f"UQ result must use float64: {results_path}:{name}")
        if not bool(np.isfinite(actual_array).all()):
            raise HardFailure(f"UQ result contains non-finite values: {results_path}:{name}")
        if not np.array_equal(actual_array, expected_array):
            raise HardFailure(f"UQ result does not match recomputation: {results_path}:{name}")

    manifest_path = publication_root / "manifest.json"
    manifest = _load_json(manifest_path)
    expected_formula = {
        "standard_deviation": "sample_ddof_1",
        "gmd": "distinct_unordered_pairs",
        "reduction_dtype": "float64",
    }
    if manifest.get("schema") != "upet.bootstrap.uncertainty/v1":
        raise HardFailure(f"unexpected UQ manifest schema: {manifest_path}")
    if manifest.get("formula") != expected_formula:
        raise HardFailure(f"unexpected UQ formula declaration: {manifest_path}")
    if (
        manifest.get("split") != split
        or manifest.get("parameter_mode") != mode
        or manifest.get("member_count") != member_count
    ):
        raise HardFailure(f"UQ manifest identity mismatch: {manifest_path}")
    results = manifest.get("results")
    if not isinstance(results, dict) or results.get("path") != "results.npz":
        raise HardFailure(f"invalid UQ results declaration: {manifest_path}")
    if results.get("sha256") != sha256_file(results_path):
        raise HardFailure(f"UQ results digest mismatch: {manifest_path}")
    declared_arrays = results.get("arrays")
    expected_arrays = {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in sorted(expected.items())
    }
    if declared_arrays != expected_arrays:
        raise HardFailure(f"UQ array declaration mismatch: {manifest_path}")
    return manifest
