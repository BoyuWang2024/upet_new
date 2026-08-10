from __future__ import annotations

from pathlib import Path

import torch

from Uncertainty_Quantification.FGE.fge import evaluation, prediction, training
from Uncertainty_Quantification.FGE.internal_migration.migration.converter import (
    convert_legacy_run,
)

from .helpers import build_conversion_case


def test_converter_never_calls_training_prediction_evaluation_or_model_compute(
    legacy_tree, config_payload, tmp_path: Path, monkeypatch
) -> None:
    source, base, config, expected = build_conversion_case(
        legacy_tree, config_payload, tmp_path
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("converter invoked forbidden compute")

    monkeypatch.setattr(torch.nn.Module, "forward", forbidden)
    monkeypatch.setattr(torch.nn.Linear, "forward", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(torch.optim.Optimizer, "step", forbidden)
    monkeypatch.setattr(torch.optim.Adam, "step", forbidden)
    monkeypatch.setattr(training, "train_fge", forbidden)
    monkeypatch.setattr(prediction, "predict_members", forbidden)
    monkeypatch.setattr(evaluation, "evaluate_fge", forbidden)

    convert_legacy_run(
        source,
        tmp_path / "published",
        tmp_path / "audit",
        config,
        base,
        expected=expected,
    )
