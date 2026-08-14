from __future__ import annotations

import numpy as np
import pytest


def test_member_samples_are_reproducible_and_member_specific() -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.sampling import (
        derive_member_seeds,
        draw_bootstrap_sample,
    )

    member_zero = derive_member_seeds(2026, 0)
    assert member_zero == derive_member_seeds(2026, 0)
    assert len(set(member_zero.as_tuple())) == 4
    member_one = derive_member_seeds(2026, 1)
    assert member_zero != member_one

    first = draw_bootstrap_sample(10, member_zero.sampling)
    repeat = draw_bootstrap_sample(10, member_zero.sampling)
    different = draw_bootstrap_sample(10, member_one.sampling)
    assert np.array_equal(first.indices, repeat.indices)
    assert not np.array_equal(first.indices, different.indices)
    assert np.array_equal(
        first.oob,
        np.setdiff1d(np.arange(10, dtype=np.int64), np.unique(first.indices)),
    )


def test_bootstrap_sample_size_is_configurable() -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.errors import HardFailure
    from Uncertainty_Quantification.BootStrapping.bootstrap.sampling import (
        draw_bootstrap_sample,
    )

    assert draw_bootstrap_sample(4, seed=5, sample_size=7).indices.shape == (7,)
    with pytest.raises(HardFailure, match="dataset_size"):
        draw_bootstrap_sample(0, seed=5)
