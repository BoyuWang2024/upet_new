"""Deterministic member seeds and bootstrap samples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .errors import HardFailure


@dataclass(frozen=True)
class MemberSeeds:
    sampling: int
    loader: int
    python: int
    torch: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.sampling, self.loader, self.python, self.torch)


@dataclass(frozen=True)
class BootstrapSample:
    indices: NDArray[np.int64]
    oob: NDArray[np.int64]
    dataset_size: int
    sample_size: int
    seed: int


def derive_member_seeds(base_seed: int, member_index: int) -> MemberSeeds:
    """Derive four independent stable seeds from run and member identity."""

    if isinstance(base_seed, bool) or base_seed < 0:
        raise HardFailure("base_seed must be a non-negative integer")
    if isinstance(member_index, bool) or member_index < 0:
        raise HardFailure("member_index must be a non-negative integer")
    children = np.random.SeedSequence([base_seed, member_index]).spawn(4)
    values = tuple(
        int(child.generate_state(1, dtype=np.uint32)[0]) for child in children
    )
    return MemberSeeds(*values)


def draw_bootstrap_sample(
    dataset_size: int, seed: int, sample_size: int | None = None
) -> BootstrapSample:
    """Draw with replacement and return sorted out-of-bag source indices."""

    if isinstance(dataset_size, bool) or dataset_size < 1:
        raise HardFailure("dataset_size must be at least 1")
    if isinstance(seed, bool) or seed < 0:
        raise HardFailure("seed must be a non-negative integer")
    size = dataset_size if sample_size is None else sample_size
    if isinstance(size, bool) or size < 1:
        raise HardFailure("sample_size must be at least 1")
    generator = np.random.Generator(np.random.PCG64(seed))
    indices = generator.integers(0, dataset_size, size=size, dtype=np.int64)
    oob = np.setdiff1d(
        np.arange(dataset_size, dtype=np.int64), np.unique(indices), assume_unique=True
    )
    indices.setflags(write=False)
    oob.setflags(write=False)
    return BootstrapSample(
        indices=indices,
        oob=oob,
        dataset_size=dataset_size,
        sample_size=size,
        seed=seed,
    )
