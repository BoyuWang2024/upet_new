from __future__ import annotations


def test_dataset_loss_is_invariant_to_batch_partition() -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.loss_metrics import (
        LossAccumulator,
        LossStatistics,
    )

    split = LossAccumulator("mean")
    split.update(
        LossStatistics({"energy": 6.0, "forces": 10.0}, {"energy": 2, "forces": 5})
    )
    split.update(
        LossStatistics({"energy": 9.0, "forces": 8.0}, {"energy": 3, "forces": 4})
    )
    joined = LossAccumulator("mean")
    joined.update(
        LossStatistics({"energy": 15.0, "forces": 18.0}, {"energy": 5, "forces": 9})
    )

    assert split.total() == joined.total()
