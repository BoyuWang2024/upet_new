"""Member artifact layout and re-audited resume decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .checkpoint import audit_checkpoint
from .errors import HardFailure


class ResumeDecision(str, Enum):
    SKIP_VALID = "skip_valid"
    RESUME_LATEST = "resume_latest"
    START_EMPTY = "start_empty"
    HARD_FAIL = "hard_fail"


@dataclass(frozen=True)
class MemberStore:
    root: Path

    def __init__(self, root: str | Path) -> None:
        object.__setattr__(self, "root", Path(root).expanduser().absolute())

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def best(self) -> Path:
        return self.checkpoints / "best.ckpt"

    @property
    def final(self) -> Path:
        return self.checkpoints / "final.ckpt"

    @property
    def latest(self) -> Path:
        return self.checkpoints / "latest.ckpt"


def decide_resume(store: MemberStore) -> ResumeDecision:
    """Re-audit member artifacts and select the only safe next action."""

    if store.final.exists():
        audit = audit_checkpoint(store.final)
        if audit.inference_ready:
            return ResumeDecision.SKIP_VALID
        raise HardFailure(f"final checkpoint is not inference ready: {store.final}")

    if store.latest.exists():
        audit = audit_checkpoint(store.latest)
        if audit.resume_ready:
            return ResumeDecision.RESUME_LATEST
        raise HardFailure(f"latest checkpoint cannot resume training: {store.latest}")

    if not store.root.exists():
        return ResumeDecision.START_EMPTY
    try:
        has_entries = next(store.root.iterdir(), None) is not None
    except OSError as error:
        raise HardFailure(
            f"could not inspect member directory {store.root}: {error}"
        ) from error
    if not has_entries:
        return ResumeDecision.START_EMPTY
    raise HardFailure(f"member directory contains partial state: {store.root}")
