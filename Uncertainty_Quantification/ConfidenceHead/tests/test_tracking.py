from __future__ import annotations

from typing import Any

import pytest

from Uncertainty_Quantification.ConfidenceHead.confidence_head.config import (
    LoggingConfig,
)
from Uncertainty_Quantification.ConfidenceHead.confidence_head.tracking import (
    WandbTracker,
)


class _FailingSummary:
    def update(self, *_: Any, **__: Any) -> None:
        raise RuntimeError("synthetic summary failure")


class _FakeRun:
    def __init__(
        self,
        *,
        failure: str | None = None,
        run_id: str = "run-abc123",
    ) -> None:
        self.id = run_id
        self.failure = failure
        self.logged: list[dict[str, int | float]] = []
        self.summary: Any = _FailingSummary() if failure == "summary" else {}
        self.finished = False
        self.finish_exit_codes: list[int] = []

    def log(self, metrics: dict[str, int | float]) -> None:
        if self.failure == "log":
            raise RuntimeError("synthetic log failure")
        self.logged.append(dict(metrics))

    def finish(self, *, exit_code: int) -> None:
        self.finish_exit_codes.append(exit_code)
        if self.failure == "finish":
            raise RuntimeError("synthetic finish failure")
        self.finished = True


class _FakeWandb:
    def __init__(
        self,
        *,
        failure: str | None = None,
        run_id: str = "run-abc123",
    ) -> None:
        self.failure = failure
        self.run = _FakeRun(failure=failure, run_id=run_id)
        self.init_kwargs: dict[str, Any] | None = None

    def init(self, **kwargs: Any) -> _FakeRun:
        self.init_kwargs = kwargs
        if self.failure == "init":
            raise RuntimeError("synthetic init failure")
        return self.run


def test_tracker_records_metrics_summary_and_finishes() -> None:
    fake_wandb = _FakeWandb()
    logging = LoggingConfig()

    tracker = WandbTracker.start(
        logging,
        run_name="demo",
        resolved_config={"profile": "smoke"},
        resume_id=None,
        wandb_module=fake_wandb,
    )
    tracker.log({"epoch": 0, "val/total_loss_ema": 1.25})
    tracker.finish(
        {"best_epoch": 0, "best_metric": 1.25, "stop_reason": "max_epochs"},
        status="success",
    )

    assert fake_wandb.init_kwargs == {
        "project": "upet-confidence-head",
        "name": "demo",
        "mode": "offline",
        "config": {"profile": "smoke"},
        "id": None,
        "resume": None,
    }
    assert tracker.run_id == "run-abc123"
    assert fake_wandb.run.logged == [
        {"epoch": 0, "val/total_loss_ema": 1.25}
    ]
    assert fake_wandb.run.summary == {
        "best_epoch": 0,
        "best_metric": 1.25,
        "stop_reason": "max_epochs",
        "status": "success",
    }
    assert fake_wandb.run.finished is True
    assert fake_wandb.run.finish_exit_codes == [0]


def test_tracker_passes_resume_identity_to_wandb() -> None:
    fake_wandb = _FakeWandb(run_id="abc123")

    tracker = WandbTracker.start(
        LoggingConfig(wandb_mode="online", wandb_project="confidence-tests"),
        run_name="resume-demo",
        resolved_config={},
        resume_id="abc123",
        wandb_module=fake_wandb,
    )

    assert tracker.run_id == "abc123"
    assert fake_wandb.init_kwargs is not None
    assert fake_wandb.init_kwargs["id"] == "abc123"
    assert fake_wandb.init_kwargs["resume"] == "allow"
    assert fake_wandb.init_kwargs["mode"] == "online"
    assert fake_wandb.init_kwargs["project"] == "confidence-tests"


def test_disabled_tracker_does_not_touch_wandb() -> None:
    fake_wandb = _FakeWandb(failure="init")

    tracker = WandbTracker.start(
        LoggingConfig(wandb=False),
        run_name="disabled",
        resolved_config={},
        resume_id=None,
        wandb_module=fake_wandb,
    )
    tracker.log({"epoch": 0})
    tracker.finish({"stop_reason": "max_epochs"}, status="success")

    assert tracker.run_id is None
    assert fake_wandb.init_kwargs is None


def test_tracker_init_failure_warns_and_returns_an_inert_tracker() -> None:
    with pytest.warns(RuntimeWarning, match="init"):
        tracker = WandbTracker.start(
            LoggingConfig(),
            run_name="init-failure",
            resolved_config={},
            resume_id=None,
            wandb_module=_FakeWandb(failure="init"),
        )

    assert tracker.run_id is None
    tracker.log({"epoch": 0})
    tracker.finish({"stop_reason": "exception"}, status="failed")


def test_tracker_log_failure_is_only_a_runtime_warning() -> None:
    tracker = WandbTracker.start(
        LoggingConfig(),
        run_name="log-failure",
        resolved_config={},
        resume_id=None,
        wandb_module=_FakeWandb(failure="log"),
    )

    with pytest.warns(RuntimeWarning, match="log"):
        tracker.log({"epoch": 0})


def test_tracker_finish_failure_is_only_a_runtime_warning() -> None:
    fake_wandb = _FakeWandb(failure="finish")
    tracker = WandbTracker.start(
        LoggingConfig(),
        run_name="finish-failure",
        resolved_config={},
        resume_id=None,
        wandb_module=fake_wandb,
    )

    with pytest.warns(RuntimeWarning, match="finish"):
        tracker.finish({"stop_reason": "exception"}, status="failed")

    assert fake_wandb.run.summary["status"] == "failed"
    assert fake_wandb.run.finish_exit_codes == [1]


def test_tracker_warns_and_disables_logging_when_resume_id_changes() -> None:
    fake_wandb = _FakeWandb(run_id="unexpected-new-id")

    with pytest.warns(RuntimeWarning, match="resume.*ID|ID.*mismatch"):
        tracker = WandbTracker.start(
            LoggingConfig(),
            run_name="resume-mismatch",
            resolved_config={},
            resume_id="stable-old-id",
            wandb_module=fake_wandb,
        )

    tracker.log({"epoch": 1})
    assert tracker.run_id == "stable-old-id"
    assert fake_wandb.run.logged == []
    assert fake_wandb.run.finish_exit_codes == [1]
    assert fake_wandb.run.finished is True


def test_resume_id_mismatch_cleanup_failure_warns_without_propagating() -> None:
    fake_wandb = _FakeWandb(failure="finish", run_id="unexpected-new-id")

    with pytest.warns(RuntimeWarning) as caught:
        tracker = WandbTracker.start(
            LoggingConfig(),
            run_name="resume-mismatch-cleanup-failure",
            resolved_config={},
            resume_id="stable-old-id",
            wandb_module=fake_wandb,
        )

    messages = [str(warning.message) for warning in caught]
    assert any("resume ID" in message for message in messages)
    assert any("finish" in message for message in messages)
    assert tracker.run_id == "stable-old-id"
    assert fake_wandb.run.finish_exit_codes == [1]


def test_tracker_failed_status_finishes_with_nonzero_exit_code() -> None:
    fake_wandb = _FakeWandb()
    tracker = WandbTracker.start(
        LoggingConfig(),
        run_name="failed-status",
        resolved_config={},
        resume_id=None,
        wandb_module=fake_wandb,
    )

    tracker.finish({"stop_reason": "exception"}, status="failed")

    assert fake_wandb.run.finish_exit_codes == [1]
    assert fake_wandb.run.finished is True


def test_tracker_summary_failure_warns_but_still_finishes() -> None:
    fake_wandb = _FakeWandb(failure="summary")
    tracker = WandbTracker.start(
        LoggingConfig(),
        run_name="summary-failure",
        resolved_config={},
        resume_id=None,
        wandb_module=fake_wandb,
    )

    with pytest.warns(RuntimeWarning, match="summary"):
        tracker.finish({"stop_reason": "exception"}, status="failed")

    assert fake_wandb.run.finish_exit_codes == [1]
    assert fake_wandb.run.finished is True
