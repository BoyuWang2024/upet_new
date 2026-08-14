from __future__ import annotations

from pathlib import Path
import random
import numpy as np

import pytest
import torch


def _save_checkpoint(
    path: Path,
    *,
    raw: dict[str, torch.Tensor] | None = None,
    ema: dict[str, torch.Tensor] | None = None,
    include_optimizer: bool = True,
) -> Path:
    raw = raw or {"head.weight": torch.arange(4, dtype=torch.float32)}
    ema = ema or {key: value + 0.5 for key, value in raw.items()}
    document = {
        "schema": "upet.bootstrap.checkpoint/v1",
        "epoch": 3,
        "validation_loss": 0.25,
        "raw_state_dict": raw,
        "ema_state_dict": ema,
    }
    if include_optimizer:
        document["optimizer_state_dict"] = {"state": {}, "param_groups": []}
        document["python_rng_state"] = random.getstate()
        document["numpy_rng_state"] = np.random.get_state()
        document["torch_rng_state"] = torch.get_rng_state()
    torch.save(document, path)
    return path


def test_checkpoint_requires_matching_raw_and_ema_keys(tmp_path: Path) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.checkpoint import (
        audit_checkpoint,
    )
    from Uncertainty_Quantification.BootStrapping.bootstrap.errors import HardFailure

    path = _save_checkpoint(
        tmp_path / "member.ckpt",
        raw={"head.weight": torch.ones(1)},
        ema={"other.weight": torch.ones(1)},
    )
    with pytest.raises(HardFailure, match="raw and EMA"):
        audit_checkpoint(path)


def test_checkpoint_capabilities_and_branch_loading(tmp_path: Path) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.checkpoint import (
        audit_checkpoint,
        load_checkpoint_branch,
    )

    path = _save_checkpoint(tmp_path / "member.ckpt", include_optimizer=False)
    audit = audit_checkpoint(path)

    assert audit.epoch == 3
    assert audit.validation_loss == 0.25
    assert audit.parameter_count == 4
    assert audit.inference_ready is True
    assert audit.resume_ready is False
    assert torch.equal(
        load_checkpoint_branch(path, "raw")["head.weight"],
        torch.arange(4, dtype=torch.float32),
    )
    assert torch.equal(
        load_checkpoint_branch(path, "ema")["head.weight"],
        torch.arange(4, dtype=torch.float32) + 0.5,
    )


def test_checkpoint_with_optimizer_is_resume_ready(tmp_path: Path) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.checkpoint import (
        audit_checkpoint,
    )

    audit = audit_checkpoint(_save_checkpoint(tmp_path / "latest.ckpt"))

    assert audit.inference_ready is True
    assert audit.resume_ready is True
