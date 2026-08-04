from __future__ import annotations

from typing import Any

import pytest
import torch

from Uncertainty_Quantification.ConfidenceHead.confidence_head.config import (
    ConfidenceConfig,
)
from Uncertainty_Quantification.ConfidenceHead.confidence_head.workflows.train import (
    train_run,
)
from Uncertainty_Quantification.ConfidenceHead.tests.test_workflows import (
    complete_cache as complete_cache,
)
from Uncertainty_Quantification.ConfidenceHead.tests.test_workflows import (
    config as config,
)


def _copy(config: ConfidenceConfig, **sections: dict[str, Any]) -> ConfidenceConfig:
    payload = config.model_dump(mode="python")
    for section, updates in sections.items():
        payload[section].update(updates)
    return ConfidenceConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("grad_clip_norm", "expected_calls"),
    [(0.25, 1), (None, 0)],
)
def test_training_applies_optional_gradient_clipping(
    config: ConfidenceConfig,
    complete_cache: Any,
    monkeypatch: pytest.MonkeyPatch,
    grad_clip_norm: float | None,
    expected_calls: int,
) -> None:
    configured = _copy(
        config,
        trainer={
            "batch_size": 16,
            "max_epochs": 1,
            "grad_clip_norm": grad_clip_norm,
        },
        logging={"wandb": False},
    )
    calls: list[float] = []
    original = torch.nn.utils.clip_grad_norm_

    def recording_clip(
        parameters: Any,
        max_norm: float,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        calls.append(max_norm)
        return original(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", recording_clip)
    train_run(
        configured,
        cache_manifest_path=complete_cache,
        run_name=f"clip-{grad_clip_norm}",
    )

    assert len(calls) == expected_calls
    if calls:
        assert calls == [pytest.approx(0.25)]


def test_step_tracking_reports_active_losses_and_throughput(
    config: ConfidenceConfig,
    complete_cache: Any,
) -> None:
    configured = _copy(
        config,
        model={"energy": {**config.model.energy.model_dump(), "enabled": False}},
        loss={"force_coefficient": 1.0, "energy_coefficient": 0.0},
        trainer={"batch_size": 16, "max_epochs": 1},
        logging={"wandb": True, "log_interval_steps": 1},
    )
    logged: list[dict[str, int | float]] = []

    class RecordingTracker:
        run_id = "step-run"

        def log(self, metrics: dict[str, int | float]) -> None:
            logged.append(dict(metrics))

        def finish(self, summary: dict[str, Any], *, status: str) -> None:
            del summary, status

    def factory(
        logging: Any,
        *,
        run_name: str,
        resolved_config: dict[str, Any],
        resume_id: str | None,
    ) -> RecordingTracker:
        del logging, run_name, resolved_config, resume_id
        return RecordingTracker()

    train_run(
        configured,
        cache_manifest_path=complete_cache,
        run_name="step-telemetry",
        tracker_factory=factory,
    )

    steps = [item for item in logged if "performance/samples_per_second" in item]
    assert len(steps) == 1
    step = steps[0]
    assert step["global_step"] == 1
    assert step["learning_rate"] == pytest.approx(configured.optimizer.learning_rate)
    assert "train/force_loss" in step
    assert "train/energy_loss" not in step
    assert "train/total_loss" in step
    assert float(step["performance/samples_per_second"]) > 0
    assert float(step["performance/atoms_per_second"]) > 0
    assert float(step["performance/data_wait_seconds"]) >= 0
    epoch = [item for item in logged if "epoch" in item][-1]
    assert "train/energy_loss" not in epoch
    assert "val/energy_loss" not in epoch
