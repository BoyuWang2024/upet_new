from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class BinningSpec:
    algorithm: str
    num_bins: int
    max_error: float
    thresholds: Tensor
    representatives: Tensor


def fixed_linear_binning(num_bins: int, max_error: float) -> BinningSpec:
    if num_bins < 3:
        raise ValueError("num_bins must be at least 3")
    if not max_error > 0:
        raise ValueError("max_error must be positive")

    width = max_error / num_bins
    thresholds = torch.arange(1, num_bins, dtype=torch.float32) * width
    representatives = (torch.arange(num_bins, dtype=torch.float32) + 0.5) * width
    return BinningSpec(
        algorithm="fixed_linear_v1",
        num_bins=num_bins,
        max_error=max_error,
        thresholds=thresholds,
        representatives=representatives,
    )


def labels_from_thresholds(values: Tensor, thresholds: Tensor) -> Tensor:
    if not torch.isfinite(values).all() or not torch.all(values >= 0):
        raise ValueError("values must be finite and non-negative")

    return torch.bucketize(
        values,
        thresholds.to(values),
        right=True,
    ).to(torch.int64)


def expected_error(logits: Tensor, representatives: Tensor) -> Tensor:
    if logits.ndim == 0 or logits.shape[-1] != representatives.numel():
        raise ValueError("logits and representatives must have matching bins")

    return torch.sum(
        torch.softmax(logits, dim=-1) * representatives.to(logits),
        dim=-1,
    )
