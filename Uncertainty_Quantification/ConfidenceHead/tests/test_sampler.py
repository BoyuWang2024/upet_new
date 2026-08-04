from __future__ import annotations

from dataclasses import dataclass

import pytest

from Uncertainty_Quantification.ConfidenceHead.confidence_head.sampler import (
    EpochRandomSampler,
    MaxAtomBatchSampler,
)


@dataclass
class AtomDataset:
    counts: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.counts)

    def num_atoms(self, index: int) -> int:
        return self.counts[index]


def test_epoch_sampler_is_complete_deterministic_and_epoch_specific() -> None:
    dataset = AtomDataset((2, 4, 1, 3, 2, 5))
    sampler = EpochRandomSampler(dataset, seed=1234)

    sampler.set_epoch(3)
    first = list(sampler)
    sampler.set_epoch(3)

    assert list(sampler) == first
    assert sorted(first) == list(range(len(dataset)))
    sampler.set_epoch(4)
    assert list(sampler) != first


def test_ordered_epoch_sampler_preserves_dataset_order() -> None:
    dataset = AtomDataset((2, 4, 1))
    sampler = EpochRandomSampler(dataset, seed=7, shuffle=False)

    sampler.set_epoch(99)

    assert list(sampler) == [0, 1, 2]


def test_max_atom_batches_respect_limit_and_cover_every_structure() -> None:
    dataset = AtomDataset((2, 4, 1, 3, 2, 5))
    sampler = MaxAtomBatchSampler(
        dataset,
        max_atoms=6,
        min_atoms=0,
        seed=4,
        shuffle=True,
        drop_last=False,
    )

    batches = list(sampler)

    assert sorted(index for batch in batches for index in batch) == list(
        range(len(dataset))
    )
    assert all(
        sum(dataset.num_atoms(index) for index in batch) <= 6 for batch in batches
    )
    sampler.set_epoch(2)
    second = list(sampler)
    sampler.set_epoch(2)
    assert list(sampler) == second


def test_ordered_atom_batches_are_stable_for_validation() -> None:
    dataset = AtomDataset((2, 4, 1, 3))
    sampler = MaxAtomBatchSampler(
        dataset,
        max_atoms=6,
        min_atoms=0,
        seed=4,
        shuffle=False,
        drop_last=False,
    )

    assert list(sampler) == [[0, 1], [2, 3]]
    assert len(sampler) == 2


def test_structure_larger_than_atom_limit_is_rejected() -> None:
    dataset = AtomDataset((2, 7, 1))

    with pytest.raises(ValueError, match="structure 1.*7.*max_atoms 6"):
        list(
            MaxAtomBatchSampler(
                dataset,
                max_atoms=6,
                min_atoms=0,
                seed=4,
                shuffle=False,
                drop_last=False,
            )
        )


def test_drop_last_uses_minimum_atom_threshold_only_for_residual() -> None:
    dataset = AtomDataset((3, 3, 1))

    kept = MaxAtomBatchSampler(
        dataset,
        max_atoms=6,
        min_atoms=2,
        seed=0,
        shuffle=False,
        drop_last=True,
    )
    complete = MaxAtomBatchSampler(
        dataset,
        max_atoms=6,
        min_atoms=2,
        seed=0,
        shuffle=False,
        drop_last=False,
    )

    assert list(kept) == [[0, 1]]
    assert list(complete) == [[0, 1], [2]]
