"""Configured dataset layout and bootstrap occurrence views."""

from __future__ import annotations

import stat
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

import numpy as np
from numpy.typing import NDArray

from .errors import HardFailure


T = TypeVar("T")


def _dataset_file(path: str | Path, location: str) -> Path:
    result = Path(path).expanduser().resolve()
    if Path(path).expanduser().is_symlink():
        raise HardFailure(f"{location} must not be a symlink: {path}")
    try:
        mode = result.stat().st_mode
    except OSError as error:
        raise HardFailure(f"{location} is not readable: {result}: {error}") from error
    if not stat.S_ISREG(mode):
        raise HardFailure(f"{location} must be a regular file: {result}")
    return result


@dataclass(frozen=True)
class DatasetLayout:
    """Resolved split paths; intentionally contains no dataset fingerprint."""

    train: Path
    val: Path
    test: Path

    def __init__(self, train: str | Path, val: str | Path, test: str | Path) -> None:
        object.__setattr__(self, "train", _dataset_file(train, "data.train"))
        object.__setattr__(self, "val", _dataset_file(val, "data.val"))
        object.__setattr__(self, "test", _dataset_file(test, "data.test"))

    def for_split(self, split: str) -> Path:
        if split not in {"train", "val", "test"}:
            raise HardFailure("split must be train, val, or test")
        return getattr(self, split)


@dataclass(frozen=True)
class Occurrence(Generic[T]):
    """One sampled occurrence of a source dataset item."""

    value: T
    source_index: int
    occurrence_index: int
    occurrence_id: str


class OccurrenceDataset(Sequence[Occurrence[T]], Generic[T]):
    """A bootstrap view that keeps duplicate source rows distinguishable."""

    def __init__(self, source: Sequence[T], indices: NDArray[np.integer]) -> None:
        array = np.asarray(indices)
        if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
            raise HardFailure(
                "bootstrap indices must be a one-dimensional integer array"
            )
        normalized = array.astype(np.int64, copy=True)
        if normalized.size and (
            int(normalized.min()) < 0 or int(normalized.max()) >= len(source)
        ):
            raise HardFailure("bootstrap index is out of bounds for source dataset")
        normalized.setflags(write=False)
        self._source = source
        self._indices = normalized

    @property
    def indices(self) -> NDArray[np.int64]:
        return self._indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> Occurrence[T]:
        source_index = int(self._indices[index])
        occurrence_index = index if index >= 0 else len(self) + index
        return Occurrence(
            value=self._source[source_index],
            source_index=source_index,
            occurrence_index=occurrence_index,
            occurrence_id=(
                f"source_{source_index:09d}_occurrence_{occurrence_index:09d}"
            ),
        )
