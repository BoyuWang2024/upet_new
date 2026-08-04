"""Verification for prediction artifacts containing one active target."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .config import ForceTargetMode
from .errors import force_error_definition


def verify_single_target_predictions(
    path: Path,
    manifest: Mapping[str, Any],
    bins: Mapping[str, Any],
    force_mode: ForceTargetMode,
) -> None:
    targets = manifest.get("active_targets")
    if targets not in (["force"], ["energy"]):
        raise ValueError("single-target evaluation active_targets mismatch")
    target = targets[0]
    predictions = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(predictions, Mapping):
        raise ValueError("test predictions must contain a mapping")
    common = {"structure_ids", "atom_offsets"}
    branch = {
        f"{target}_logits",
        f"{target}_labels",
        f"{target}_observed_errors",
        f"{target}_expected_errors",
        f"{target}_representatives",
    }
    semantics = (
        {"force_target_mode", "force_error_definition"} if target == "force" else set()
    )
    if set(predictions) != common | branch | semantics:
        raise ValueError("single-target prediction fields mismatch")
    if not all(isinstance(predictions[name], torch.Tensor) for name in common | branch):
        raise ValueError("single-target prediction fields must be tensors")
    ids = predictions["structure_ids"]
    offsets = predictions["atom_offsets"]
    counts = manifest.get("test_counts")
    if (
        ids.dtype != torch.int64
        or offsets.dtype != torch.int64
        or ids.ndim != 1
        or offsets.shape != (len(ids) + 1,)
        or int(offsets[0]) != 0
        or not bool(torch.all(offsets[1:] > offsets[:-1]))
        or not isinstance(counts, Mapping)
        or counts.get("structures") != len(ids)
        or counts.get("atoms") != int(offsets[-1])
        or counts.get("force_components") != 3 * int(offsets[-1])
    ):
        raise ValueError("single-target prediction counts are inconsistent")
    logits = predictions[f"{target}_logits"]
    labels = predictions[f"{target}_labels"]
    observed = predictions[f"{target}_observed_errors"]
    expected = predictions[f"{target}_expected_errors"]
    representatives = predictions[f"{target}_representatives"]
    if (
        not logits.is_floating_point()
        or labels.dtype != torch.int64
        or not observed.is_floating_point()
        or not expected.is_floating_point()
        or representatives.ndim != 1
        or len(representatives) != logits.shape[-1]
        or observed.shape != labels.shape
        or expected.shape != labels.shape
    ):
        raise ValueError("single-target prediction shapes or dtypes mismatch")
    if target == "force":
        expected_logits_ndim = 2 if force_mode == "atom_mean" else 3
        expected_labels_ndim = 1 if force_mode == "atom_mean" else 2
        if logits.ndim != expected_logits_ndim or labels.ndim != expected_labels_ndim:
            raise ValueError("force prediction shapes are inconsistent")
        if labels.shape[0] != int(offsets[-1]):
            raise ValueError("force target count disagrees with atom count")
        if (
            predictions.get("force_target_mode") != force_mode
            or predictions.get("force_error_definition")
            != force_error_definition(force_mode)
            or counts.get("force_targets") != labels.numel()
            or counts.get("force_target_mode") != force_mode
        ):
            raise ValueError("force target semantics mismatch")
    elif logits.ndim != 2 or labels.ndim != 1 or len(labels) != len(ids):
        raise ValueError("energy prediction shapes are inconsistent")
    try:
        declared = torch.tensor(
            bins[target]["representatives"], dtype=representatives.dtype
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("binning representatives are invalid") from error
    if not torch.equal(representatives, declared):
        raise ValueError("prediction representatives disagree with binning artifact")
    if not bool(torch.isfinite(logits).all()) or not bool(
        torch.isfinite(observed).all()
    ):
        raise ValueError("single-target predictions must be finite")
    if not bool(torch.all((labels >= 0) & (labels < logits.shape[-1]))):
        raise ValueError("single-target labels are out of range")
