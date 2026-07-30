"""Numerically explicit metrics for confidence classification."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F

from .binning import expected_error


def _ranks(values: Tensor) -> Tensor:
    """Return average, zero-based ranks, including ties."""
    flat = values.reshape(-1)
    order = torch.argsort(flat, stable=True)
    sorted_values = flat[order]
    ranks = torch.empty(len(flat), dtype=torch.float64, device=flat.device)
    start = 0
    while start < len(flat):
        stop = start + 1
        while stop < len(flat) and bool(sorted_values[stop] == sorted_values[start]):
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def _correlation(left: Tensor, right: Tensor) -> float:
    left = left.reshape(-1).to(torch.float64)
    right = right.reshape(-1).to(torch.float64)
    left = left - left.mean()
    right = right - right.mean()
    denominator = torch.sqrt(left.square().sum() * right.square().sum())
    if not bool(denominator > 0):
        return 0.0
    result = float((left * right).sum() / denominator)
    return result if math.isfinite(result) else 0.0


def _overflow_boundary(representatives: Tensor) -> Tensor:
    representatives = representatives.reshape(-1)
    if len(representatives) < 2:
        raise ValueError("at least two bin representatives are required")
    widths = representatives[1:] - representatives[:-1]
    if not bool(torch.all(widths > 0)):
        raise ValueError("representatives must be strictly increasing")
    return representatives[-1] + widths[-1] / 2


def classification_metrics(
    logits: Tensor,
    labels: Tensor,
    observed: Tensor,
    representatives: Tensor,
) -> dict[str, int | float]:
    """Compute classification and continuous-error diagnostics."""
    if logits.ndim != 2 or logits.shape[1] != representatives.numel():
        raise ValueError("logits must have shape [N, B] matching representatives")
    labels = labels.reshape(-1)
    observed = observed.reshape(-1)
    if len(logits) == 0 or labels.shape != observed.shape or len(labels) != len(logits):
        raise ValueError("logits, labels, and observed errors must have matching N")
    if labels.dtype != torch.int64:
        raise ValueError("labels must have dtype int64")
    if (
        not bool(torch.isfinite(logits).all())
        or not bool(torch.isfinite(observed).all())
        or not bool(torch.all(observed >= 0))
    ):
        raise ValueError("logits and observed errors must be finite and non-negative")
    if bool(torch.any(labels < 0)) or bool(torch.any(labels >= logits.shape[1])):
        raise ValueError("labels are outside the configured bins")

    probabilities = torch.softmax(logits, dim=-1)
    expected = expected_error(logits, representatives)
    one_hot = F.one_hot(labels, num_classes=logits.shape[-1]).to(probabilities)
    nll = F.cross_entropy(logits, labels)
    brier = (probabilities - one_hot).square().sum(dim=-1).mean()
    overflow_count = int(
        (observed > _overflow_boundary(representatives).to(observed)).sum().item()
    )
    count = len(labels)
    return {
        "sample_count": count,
        "accuracy": float((logits.argmax(dim=-1) == labels).float().mean()),
        "nll": float(nll),
        "brier": float(brier),
        "mean_observed_error": float(observed.mean()),
        "mean_expected_error": float(expected.mean()),
        "mae_expected_vs_observed": float(torch.abs(expected - observed).mean()),
        "pearson": _correlation(expected, observed),
        "spearman": _correlation(_ranks(expected), _ranks(observed)),
        "overflow_count": overflow_count,
        "overflow_fraction": overflow_count / count,
    }
