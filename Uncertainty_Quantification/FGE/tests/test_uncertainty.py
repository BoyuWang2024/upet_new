"""Regression tests for the published legacy FGE uncertainty formula."""

from __future__ import annotations

import math

import pytest
import torch

from Uncertainty_Quantification.FGE.fge.uncertainty import (
    FORMULA_VERSION,
    population_std,
    reduce_force_by_structure,
    scalar_gmd,
    vector_gmd,
)


def test_population_std_is_population_not_sample_and_preserves_dtype() -> None:
    members = torch.tensor([[0.0, 2.0], [2.0, 4.0]], dtype=torch.float64)

    result = population_std(members)

    assert FORMULA_VERSION == "legacy_upet_fge_v1"
    assert result.dtype == torch.float64
    torch.testing.assert_close(result, torch.tensor([1.0, 1.0], dtype=torch.float64))
    assert not torch.allclose(result, torch.std(members, dim=0, unbiased=True))


def test_population_std_supports_an_explicit_member_dimension() -> None:
    members = torch.tensor([[0.0, 2.0], [2.0, 4.0]], dtype=torch.float32)

    result = population_std(members, dim=1)

    torch.testing.assert_close(result, torch.tensor([1.0, 1.0]))


def test_scalar_gmd_uses_ordered_k_squared_pairs_including_self_pairs() -> None:
    members = torch.tensor([0.0, 2.0], dtype=torch.float64)

    result = scalar_gmd(members)

    assert result == torch.tensor(1.0, dtype=torch.float64)
    assert result != torch.tensor(2.0, dtype=torch.float64)  # unordered i < j mean


def test_scalar_gmd_handles_three_members_elementwise() -> None:
    members = torch.tensor([[0.0, 0.0], [3.0, 6.0], [6.0, 12.0]], dtype=torch.float32)

    result = scalar_gmd(members)

    torch.testing.assert_close(result, torch.tensor([8.0 / 3.0, 16.0 / 3.0]))


def test_vector_gmd_uses_l2_distance_then_ordered_k_squared_mean() -> None:
    members = torch.tensor([[[0.0, 0.0, 0.0]], [[3.0, 4.0, 0.0]]], dtype=torch.float64)

    result = vector_gmd(members)

    torch.testing.assert_close(result, torch.tensor([2.5], dtype=torch.float64))
    assert result.dtype == members.dtype


def test_force_structure_reductions_preserve_structure_order_and_dtype() -> None:
    values = torch.tensor([1.0, 3.0, 2.0, 8.0, 10.0], dtype=torch.float64)
    offsets = torch.tensor([0, 2, 5], dtype=torch.int64)

    result = reduce_force_by_structure(values, offsets, quantile=0.95)

    assert set(result) == {"mean", "max", "q95"}
    torch.testing.assert_close(
        result["mean"], torch.tensor([2.0, 20.0 / 3.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        result["max"], torch.tensor([3.0, 10.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        result["q95"], torch.tensor([2.9, 9.8], dtype=torch.float64)
    )
    assert all(value.dtype == torch.float64 for value in result.values())


@pytest.mark.parametrize(
    "operation,members",
    [
        (population_std, torch.tensor([1.0, math.nan])),
        (scalar_gmd, torch.tensor([1.0, math.inf])),
        (vector_gmd, torch.tensor([[1.0, 2.0], [3.0, -math.inf]])),
    ],
)
def test_member_math_rejects_non_finite_inputs(operation, members) -> None:
    with pytest.raises(ValueError, match="finite"):
        operation(members)


@pytest.mark.parametrize(
    "operation,members",
    [
        (population_std, torch.tensor([1.0])),
        (scalar_gmd, torch.tensor([1.0])),
        (vector_gmd, torch.tensor([[1.0, 2.0]])),
    ],
)
def test_member_math_rejects_fewer_than_two_members(operation, members) -> None:
    with pytest.raises(ValueError, match="members"):
        operation(members)


def test_member_math_rejects_invalid_tensor_ranks_and_dtypes() -> None:
    with pytest.raises(ValueError, match="rank"):
        scalar_gmd(torch.tensor(1.0))
    with pytest.raises(ValueError, match="rank"):
        vector_gmd(torch.tensor([1.0, 2.0]))
    with pytest.raises(TypeError, match="floating"):
        population_std(torch.tensor([1, 2]))


@pytest.mark.parametrize(
    "values,offsets",
    [
        (torch.tensor([[1.0]]), torch.tensor([0, 1])),
        (torch.tensor([1.0, 2.0]), torch.tensor([[0, 2]])),
        (torch.tensor([1.0, 2.0]), torch.tensor([1, 2])),
        (torch.tensor([1.0, 2.0]), torch.tensor([0, 1])),
        (torch.tensor([1.0, 2.0]), torch.tensor([0, 2, 1, 2])),
        (torch.tensor([1.0, 2.0]), torch.tensor([0, 1, 1, 2])),
        (torch.tensor([]), torch.tensor([0, 0])),
    ],
)
def test_force_structure_reduction_rejects_invalid_or_empty_partitions(
    values: torch.Tensor, offsets: torch.Tensor
) -> None:
    with pytest.raises((TypeError, ValueError), match="values|offsets|structure"):
        reduce_force_by_structure(values, offsets)


@pytest.mark.parametrize("quantile", [-0.1, 1.1, math.nan, math.inf])
def test_force_structure_reduction_rejects_invalid_quantiles(quantile: float) -> None:
    with pytest.raises(ValueError, match="quantile"):
        reduce_force_by_structure(
            torch.tensor([1.0, 2.0]), torch.tensor([0, 2]), quantile=quantile
        )


def test_force_structure_reduction_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        reduce_force_by_structure(torch.tensor([1.0, math.nan]), torch.tensor([0, 2]))
