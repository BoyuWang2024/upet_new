"""Bounded-memory carnet-style ensemble uncertainty reductions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
from numpy.typing import NDArray

from .errors import HardFailure


PredictionField = Literal["energy", "forces", "stress"]


@dataclass(frozen=True)
class StreamingMeanStd:
    mean: NDArray[np.float64]
    std: NDArray[np.float64]
    member_count: int


def _paths(member_paths: Iterable[str | Path]) -> tuple[Path, ...]:
    paths = tuple(Path(path).expanduser().resolve() for path in member_paths)
    if len(paths) < 2:
        raise HardFailure("uncertainty requires at least two members")
    if len(set(paths)) != len(paths):
        raise HardFailure("uncertainty member paths must be distinct")
    return paths


def _load_field(path: Path, field: str) -> NDArray[np.float64]:
    if field not in {"energy", "forces", "stress"}:
        raise HardFailure("uncertainty field must be energy, forces, or stress")
    try:
        with np.load(path, allow_pickle=False) as archive:
            if field not in archive.files:
                raise HardFailure(f"prediction artifact is missing {field}: {path}")
            result = np.asarray(archive[field], dtype=np.float64)
    except HardFailure:
        raise
    except (OSError, ValueError) as error:
        raise HardFailure(
            f"could not load prediction field {field}: {path}: {error}"
        ) from error
    if not bool(np.isfinite(result).all()):
        raise HardFailure(
            f"prediction field contains non-finite values: {path}:{field}"
        )
    return result


def streaming_mean_std(
    member_paths: Iterable[str | Path], field: PredictionField
) -> StreamingMeanStd:
    """Compute float64 sample standard deviation with Welford's algorithm."""

    paths = _paths(member_paths)
    mean: NDArray[np.float64] | None = None
    sum_squares: NDArray[np.float64] | None = None
    for count, path in enumerate(paths, start=1):
        values = _load_field(path, field)
        if mean is None:
            mean = np.array(values, dtype=np.float64, copy=True)
            sum_squares = np.zeros_like(mean)
            continue
        if values.shape != mean.shape:
            raise HardFailure(f"uncertainty {field} member layout mismatch: {path}")
        delta = values - mean
        mean += delta / count
        assert sum_squares is not None
        sum_squares += delta * (values - mean)
    assert mean is not None and sum_squares is not None
    variance = np.maximum(sum_squares / (len(paths) - 1), 0.0)
    return StreamingMeanStd(mean=mean, std=np.sqrt(variance), member_count=len(paths))


def pairwise_gmd(
    member_paths: Iterable[str | Path], field: PredictionField
) -> NDArray[np.float64]:
    """Compute mean absolute difference over unordered distinct member pairs."""

    paths = _paths(member_paths)
    total: NDArray[np.float64] | None = None
    pair_count = 0
    for left_index, left_path in enumerate(paths[:-1]):
        left = _load_field(left_path, field)
        if total is None:
            total = np.zeros_like(left, dtype=np.float64)
        elif left.shape != total.shape:
            raise HardFailure(
                f"uncertainty {field} member layout mismatch: {left_path}"
            )
        for right_path in paths[left_index + 1 :]:
            right = _load_field(right_path, field)
            if right.shape != left.shape:
                raise HardFailure(
                    f"uncertainty {field} member layout mismatch: {right_path}"
                )
            total += np.abs(left - right)
            pair_count += 1
    assert total is not None and pair_count > 0
    return total / pair_count


def scalar_rms_reductions(
    values: NDArray,
    field: PredictionField,
    *,
    num_atoms: NDArray,
    atom_offsets: NDArray,
) -> NDArray[np.float64]:
    """Reduce component UQ to the public scalar energy/vector/tensor measures."""

    array = np.asarray(values, dtype=np.float64)
    counts = np.asarray(num_atoms, dtype=np.int64)
    offsets = np.asarray(atom_offsets, dtype=np.int64)
    expected_offsets = np.concatenate(
        [np.array([0], dtype=np.int64), np.cumsum(counts, dtype=np.int64)]
    )
    if not np.array_equal(offsets, expected_offsets):
        raise HardFailure("atom_offsets do not match num_atoms")
    if field == "energy":
        if array.shape != counts.shape:
            raise HardFailure("energy reduction layout mismatch")
        return array / counts
    if field == "forces":
        if array.shape != (int(offsets[-1]), 3):
            raise HardFailure("forces reduction layout mismatch")
        return np.sqrt(np.mean(np.square(array), axis=1))
    if field == "stress":
        if array.shape != (len(counts), 3, 3):
            raise HardFailure("stress reduction layout mismatch")
        return np.sqrt(np.mean(np.square(array), axis=(1, 2)))
    raise HardFailure("uncertainty field must be energy, forces, or stress")


from .uq_publication import UncertaintyPublication, compute_store_uncertainty  # noqa: E402
