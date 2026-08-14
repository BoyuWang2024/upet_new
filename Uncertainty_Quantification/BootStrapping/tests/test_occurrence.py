from __future__ import annotations

from collections import namedtuple

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from Uncertainty_Quantification.BootStrapping.bootstrap.occurrence import (
    BootstrapOccurrenceDataset,
)

Sample = namedtuple("Sample", ["system", "energy", "forces"])


def _map(block: TensorBlock) -> TensorMap:
    return TensorMap(Labels.single(), [block])


def _sample() -> Sample:
    properties = Labels.range("energy", 1)
    xyz = Labels.range("xyz", 3)
    energy = TensorBlock(
        values=torch.tensor([[2.5]], dtype=torch.float64),
        samples=Labels(["system"], torch.tensor([[7]])),
        components=[],
        properties=properties,
    )
    energy.add_gradient(
        "positions",
        TensorBlock(
            values=torch.arange(6, dtype=torch.float64).reshape(2, 3, 1),
            samples=Labels(
                ["sample", "system", "atom"],
                torch.tensor([[0, 7, 0], [0, 7, 1]]),
            ),
            components=[xyz],
            properties=properties,
        ),
    )
    forces = TensorBlock(
        values=torch.arange(6, dtype=torch.float64).reshape(2, 3, 1),
        samples=Labels(["system", "atom"], torch.tensor([[7, 0], [7, 1]])),
        components=[xyz],
        properties=properties,
    )
    return Sample(object(), _map(energy), _map(forces))


def test_duplicate_source_rows_receive_distinct_occurrence_ids() -> None:
    source = _sample()
    dataset = BootstrapOccurrenceDataset([source], [0, 0])

    first, second = dataset[0], dataset[1]
    assert first.energy.block().samples.values.tolist() == [[0]]
    assert second.energy.block().samples.values.tolist() == [[1]]
    assert first.energy.block().gradient("positions").samples.values.tolist() == [
        [0, 0, 0],
        [0, 0, 1],
    ]
    assert second.energy.block().gradient("positions").samples.values.tolist() == [
        [0, 1, 0],
        [0, 1, 1],
    ]
    assert second.forces.block().samples.values.tolist() == [[1, 0], [1, 1]]
    assert torch.equal(second.energy.block().values, source.energy.block().values)
