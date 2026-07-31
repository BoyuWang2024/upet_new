"""Failure-tolerant Weights & Biases tracking for confidence training."""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from .config import LoggingConfig


class Tracker(Protocol):
    """Training-side tracking interface."""

    run_id: str | None

    def log(self, metrics: dict[str, int | float]) -> None: ...

    def finish(self, summary: dict[str, Any], *, status: str) -> None: ...


TrackerFactory = Callable[..., Tracker]


def _warn(stage: str, error: Exception) -> None:
    warnings.warn(
        f"W&B {stage} failed: {error}",
        RuntimeWarning,
        stacklevel=3,
    )


@dataclass
class WandbTracker:
    """Small W&B boundary that never makes remote tracking authoritative."""

    _run: Any | None
    run_id: str | None

    @classmethod
    def start(
        cls,
        logging: LoggingConfig,
        *,
        run_name: str,
        resolved_config: Mapping[str, Any],
        resume_id: str | None,
        wandb_module: Any | None = None,
    ) -> WandbTracker:
        """Start a run, returning an inert tracker when W&B is unavailable."""
        if not logging.wandb:
            return cls(_run=None, run_id=resume_id)
        try:
            module = (
                importlib.import_module("wandb")
                if wandb_module is None
                else wandb_module
            )
            run = module.init(
                project=logging.wandb_project,
                name=run_name,
                mode=logging.wandb_mode,
                config=resolved_config,
                id=resume_id,
                resume="allow" if resume_id is not None else None,
            )
        except Exception as error:
            _warn("init", error)
            return cls(_run=None, run_id=resume_id)
        raw_run_id = getattr(run, "id", None)
        sdk_run_id = str(raw_run_id) if raw_run_id is not None else None
        if resume_id is not None:
            if sdk_run_id != resume_id:
                _warn(
                    "resume ID validation",
                    RuntimeError(
                        f"SDK returned mismatched run ID {sdk_run_id!r}; "
                        f"expected {resume_id!r}"
                    ),
                )
                return cls(_run=None, run_id=resume_id)
            return cls(_run=run, run_id=resume_id)
        return cls(_run=run, run_id=sdk_run_id)

    def log(self, metrics: dict[str, int | float]) -> None:
        """Record one already-committed local epoch record."""
        if self._run is None:
            return
        try:
            self._run.log(dict(metrics))
        except Exception as error:
            _warn("log", error)

    def finish(self, summary: dict[str, Any], *, status: str) -> None:
        """Publish summary metadata and safely finish the W&B run."""
        if self._run is None:
            return
        try:
            self._run.summary.update({**summary, "status": status})
        except Exception as error:
            _warn("finish summary", error)
        try:
            self._run.finish(exit_code=0 if status == "success" else 1)
        except Exception as error:
            _warn("finish", error)
