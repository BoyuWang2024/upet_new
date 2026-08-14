from __future__ import annotations

from pathlib import Path

import pytest
import torch

from test_checkpoint import _save_checkpoint


def test_member_resume_decisions(tmp_path: Path) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.members import (
        MemberStore,
        ResumeDecision,
        decide_resume,
    )

    empty = MemberStore(tmp_path / "member_000")
    assert decide_resume(empty) is ResumeDecision.START_EMPTY

    resumable = MemberStore(tmp_path / "member_001")
    resumable.checkpoints.mkdir(parents=True)
    _save_checkpoint(resumable.latest)
    assert decide_resume(resumable) is ResumeDecision.RESUME_LATEST

    complete = MemberStore(tmp_path / "member_002")
    complete.checkpoints.mkdir(parents=True)
    _save_checkpoint(complete.final, include_optimizer=False)
    assert decide_resume(complete) is ResumeDecision.SKIP_VALID


def test_member_rejects_inference_only_latest_and_partial_state(tmp_path: Path) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.errors import HardFailure
    from Uncertainty_Quantification.BootStrapping.bootstrap.members import (
        MemberStore,
        decide_resume,
    )

    inference_only = MemberStore(tmp_path / "member_000")
    inference_only.checkpoints.mkdir(parents=True)
    _save_checkpoint(inference_only.latest, include_optimizer=False)
    with pytest.raises(HardFailure, match="cannot resume"):
        decide_resume(inference_only)

    partial = MemberStore(tmp_path / "member_001")
    partial.root.mkdir(parents=True)
    (partial.root / "unexpected.tmp").write_text("partial", encoding="utf-8")
    with pytest.raises(HardFailure, match="partial"):
        decide_resume(partial)
