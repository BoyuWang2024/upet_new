"""Publication of canonical uncertainty artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from .artifacts import atomic_write_json, atomic_write_npz, sha256_file
from .errors import HardFailure
from .prediction import load_target_arrays


@dataclass(frozen=True)
class UncertaintyPublication:
    results_path: Path
    manifest_path: Path
    split: str
    mode: str
    member_count: int


def compute_store_uncertainty(
    prediction_root: str | Path,
    output_root: str | Path,
    *,
    split: str,
    mode: str,
    member_count: int,
    units: Mapping[str, str],
) -> UncertaintyPublication:
    """Compute and publish UQ from one canonical split/mode prediction store."""

    from .uncertainty import pairwise_gmd, scalar_rms_reductions, streaming_mean_std

    if split not in {"val", "test"}:
        raise HardFailure("uncertainty split must be val or test")
    if mode not in {"raw", "ema"}:
        raise HardFailure("uncertainty mode must be raw or ema")
    if isinstance(member_count, bool) or member_count < 2:
        raise HardFailure("uncertainty member_count must be at least 2")
    prediction_path = Path(prediction_root).expanduser().resolve()
    targets = load_target_arrays(prediction_path / split / "targets.npz")
    members = tuple(
        prediction_path / split / "members" / f"member_{index:03d}" / f"{mode}.npz"
        for index in range(member_count)
    )
    if any(not path.is_file() for path in members):
        raise HardFailure(f"uncertainty requires every configured {mode} member")

    results: dict[str, NDArray[np.float64]] = {}
    for field in ("energy", "forces", "stress"):
        statistics = streaming_mean_std(members, field)
        gmd = pairwise_gmd(members, field)
        results[f"{field}_mean"] = statistics.mean
        results[f"{field}_std"] = statistics.std
        results[f"{field}_gmd"] = gmd
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

    publication_root = Path(output_root).expanduser().resolve() / split / mode
    results_path = atomic_write_npz(publication_root / "results.npz", **results)
    manifest_path = atomic_write_json(
        publication_root / "manifest.json",
        {
            "schema": "upet.bootstrap.uncertainty/v1",
            "formula": {
                "standard_deviation": "sample_ddof_1",
                "gmd": "distinct_unordered_pairs",
                "reduction_dtype": "float64",
            },
            "split": split,
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
        },
    )
    return UncertaintyPublication(
        results_path=results_path,
        manifest_path=manifest_path,
        split=split,
        mode=mode,
        member_count=member_count,
    )
