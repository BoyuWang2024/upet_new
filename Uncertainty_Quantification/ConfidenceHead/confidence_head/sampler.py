"""Deterministic epoch and atom-count samplers for memmap datasets."""

from __future__ import annotations

from collections.abc import Iterator, Sized
from typing import Protocol

import torch
from torch.utils.data import Sampler


class AtomCountDataset(Sized, Protocol):
    def num_atoms(self, index: int) -> int: ...


class EpochRandomSampler(Sampler[int]):
    """Generate one complete, epoch-addressable global permutation."""

    def __init__(
        self,
        dataset: Sized,
        *,
        seed: int,
        shuffle: bool = True,
    ) -> None:
        self.dataset = dataset
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        size = len(self.dataset)
        if not self.shuffle:
            return iter(range(size))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(size, generator=generator).tolist())

    def __len__(self) -> int:
        return len(self.dataset)


class MaxAtomBatchSampler(Sampler[list[int]]):
    """Pack a deterministic structure order into bounded-atom batches."""

    def __init__(
        self,
        dataset: AtomCountDataset,
        *,
        max_atoms: int,
        min_atoms: int,
        seed: int,
        shuffle: bool,
        drop_last: bool,
    ) -> None:
        if max_atoms <= 0:
            raise ValueError("max_atoms must be positive")
        if min_atoms < 0 or min_atoms > max_atoms:
            raise ValueError("min_atoms must be between zero and max_atoms")
        self.dataset = dataset
        self.max_atoms = int(max_atoms)
        self.min_atoms = int(min_atoms)
        self.drop_last = bool(drop_last)
        self.order = EpochRandomSampler(dataset, seed=seed, shuffle=shuffle)

    @property
    def epoch(self) -> int:
        return self.order.epoch

    def set_epoch(self, epoch: int) -> None:
        self.order.set_epoch(epoch)

    def _batches(self) -> list[list[int]]:
        batches: list[list[int]] = []
        current: list[int] = []
        current_atoms = 0
        for index in self.order:
            atoms = int(self.dataset.num_atoms(index))
            if atoms <= 0:
                raise ValueError(
                    f"structure {index} has nonpositive atom count {atoms}"
                )
            if atoms > self.max_atoms:
                raise ValueError(
                    f"structure {index} has {atoms} atoms, exceeding "
                    f"max_atoms {self.max_atoms}"
                )
            if current and current_atoms + atoms > self.max_atoms:
                batches.append(current)
                current = []
                current_atoms = 0
            current.append(index)
            current_atoms += atoms
        if current and (
            not self.drop_last or self.min_atoms == 0 or current_atoms >= self.min_atoms
        ):
            batches.append(current)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self._batches())

    def __len__(self) -> int:
        return len(self._batches())
