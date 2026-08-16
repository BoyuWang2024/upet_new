"""Numerical reduction and publication of canonical uncertainty artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from .artifacts import (
    _absolute_lexical,
    _reject_symlink_components,
    atomic_write_json,
    atomic_write_npz,
    sha256_file,
)
from .errors import HardFailure
from .identifiers import validate_artifact_key
from .prediction import load_target_arrays


_FORMULA = {
    "standard_deviation": "sample_ddof_1",
    "gmd": "distinct_unordered_pairs",
    "reduction_dtype": "float64",
}


@dataclass(frozen=True)
class UncertaintyPublication:
    results_path: Path
    manifest_path: Path
    split: str
    mode: str
    member_count: int


def _validate_request(dataset_key: object, mode: object, member_count: object) -> str:
    key = validate_artifact_key(dataset_key, "uncertainty dataset key")
    if mode not in {"raw", "ema"}:
        raise HardFailure("uncertainty mode must be raw or ema")
    if (
        isinstance(member_count, bool)
        or not isinstance(member_count, int)
        or member_count < 2
    ):
        raise HardFailure("uncertainty member_count must be at least 2")
    return key


def _member_paths(
    split_root: Path, *, mode: str, member_count: int
) -> tuple[Path, ...]:
    members = tuple(
        split_root / "members" / f"member_{index:03d}" / f"{mode}.npz"
        for index in range(member_count)
    )
    for path in members:
        _reject_symlink_components(path)
    if any(not path.is_file() for path in members):
        raise HardFailure(f"uncertainty requires every configured {mode} member")
    return members


def compute_uncertainty_results(
    prediction_split_root: str | Path, *, mode: str, member_count: int
) -> dict[str, NDArray[np.float64]]:
    """Compute canonical UQ arrays from an already-selected prediction split."""

    from .uncertainty import pairwise_gmd, scalar_rms_reductions, streaming_mean_std

    _validate_request("selected_split", mode, member_count)
    split_path = _absolute_lexical(prediction_split_root)
    _reject_symlink_components(split_path)
    split_root = split_path.resolve()
    targets_path = split_root / "targets.npz"
    _reject_symlink_components(targets_path)
    targets = load_target_arrays(targets_path)
    members = _member_paths(split_root, mode=mode, member_count=member_count)
    results: dict[str, NDArray[np.float64]] = {}
    for field in ("energy", "forces", "stress"):
        statistics = streaming_mean_std(members, field)
        results[f"{field}_mean"] = statistics.mean
        results[f"{field}_std"] = statistics.std
        results[f"{field}_gmd"] = pairwise_gmd(members, field)
    for measure in ("std", "gmd"):
        results[f"energy_per_atom_{measure}"] = scalar_rms_reductions(
            results[f"energy_{measure}"],
            "energy",
            num_atoms=targets.num_atoms,
            atom_offsets=targets.atom_offsets,
        )
        results[f"force_vector_rms_{measure}"] = scalar_rms_reductions(
            results[f"forces_{measure}"],
            "forces",
            num_atoms=targets.num_atoms,
            atom_offsets=targets.atom_offsets,
        )
        results[f"stress_tensor_rms_{measure}"] = scalar_rms_reductions(
            results[f"stress_{measure}"],
            "stress",
            num_atoms=targets.num_atoms,
            atom_offsets=targets.atom_offsets,
        )
    return results


def _manifest_document(
    results_path: Path,
    results: Mapping[str, NDArray[np.float64]],
    *,
    dataset_key: str,
    mode: str,
    member_count: int,
    units: Mapping[str, str],
) -> dict[str, object]:
    """Build the complete deterministic v1 UQ manifest document."""

    return {
        "schema": "upet.bootstrap.uncertainty/v1",
        "formula": _FORMULA,
        "split": dataset_key,
        "parameter_mode": mode,
        "member_count": member_count,
        "units": dict(units),
        "results": {
            "path": "results.npz",
            "sha256": sha256_file(results_path),
            "arrays": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in sorted(results.items())
            },
        },
    }


def publish_uncertainty_results(
    publication_root: str | Path,
    results: Mapping[str, NDArray[np.float64]],
    *,
    dataset_key: str,
    mode: str,
    member_count: int,
    units: Mapping[str, str],
) -> UncertaintyPublication:
    """Publish already-reduced UQ arrays, with the manifest written last."""

    key = _validate_request(dataset_key, mode, member_count)
    root_path = _absolute_lexical(publication_root)
    _reject_symlink_components(root_path)
    root = root_path.resolve()
    canonical = {
        name: np.asarray(value, dtype=np.float64) for name, value in results.items()
    }
    results_path = atomic_write_npz(root / "results.npz", **canonical)
    manifest_path = atomic_write_json(
        root / "manifest.json",
        _manifest_document(
            results_path,
            canonical,
            dataset_key=key,
            mode=mode,
            member_count=member_count,
            units=units,
        ),
    )
    return UncertaintyPublication(
        results_path=results_path,
        manifest_path=manifest_path,
        split=key,
        mode=mode,
        member_count=member_count,
    )


def compute_store_uncertainty(
    prediction_root: str | Path,
    output_root: str | Path,
    *,
    split: str,
    mode: str,
    member_count: int,
    units: Mapping[str, str],
) -> UncertaintyPublication:
    """Compute and publish UQ from one canonical prediction dataset."""

    key = _validate_request(split, mode, member_count)
    prediction_path = _absolute_lexical(prediction_root)
    _reject_symlink_components(prediction_path)
    prediction_path = prediction_path.resolve()
    output_path = _absolute_lexical(output_root)
    _reject_symlink_components(output_path)
    output_path = output_path.resolve()
    results = compute_uncertainty_results(
        prediction_path / key, mode=mode, member_count=member_count
    )
    return publish_uncertainty_results(
        output_path / key / mode,
        results,
        dataset_key=key,
        mode=mode,
        member_count=member_count,
        units=units,
    )
