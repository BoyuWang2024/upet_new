from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def _canonical_checkpoint(path: Path) -> Path:
    raw = {"head.weight": torch.arange(4, dtype=torch.float32)}
    torch.save(
        {
            "schema": "upet.bootstrap.checkpoint/v1",
            "epoch": 3,
            "validation_loss": 0.25,
            "raw_state_dict": raw,
            "ema_state_dict": {"head.weight": raw["head.weight"] + 0.5},
            "optimizer_state_dict": {"state": {}, "param_groups": []},
            "python_rng_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
        },
        path,
    )
    return path


def test_resume_requires_numpy_rng_state(tmp_path: Path) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.checkpoint import (
        audit_checkpoint,
    )

    assert (
        audit_checkpoint(_canonical_checkpoint(tmp_path / "latest.pt")).resume_ready
        is False
    )


def test_legacy_checkpoint_is_audited_and_loaded_without_rewriting(
    tmp_path: Path,
) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.checkpoint import (
        audit_checkpoint,
        load_checkpoint_branch,
    )

    path = tmp_path / "best.pt"
    raw = {"head.weight": torch.arange(4, dtype=torch.float32)}
    torch.save(
        {
            "schema_version": 1,
            "member_index": 1,
            "epoch": 3,
            "validation_loss": 0.25,
            "raw_state": raw,
            "ema_state": {"head.weight": raw["head.weight"] + 0.5},
        },
        path,
    )

    assert audit_checkpoint(path).inference_ready is True
    assert torch.equal(
        load_checkpoint_branch(path, "raw")["head.weight"], raw["head.weight"]
    )


class _Runtime:
    def __init__(self) -> None:
        self.model = torch.nn.Linear(1, 1, bias=False)

    def train_epoch(self, optimizer: torch.optim.Optimizer, epoch: int) -> float:
        optimizer.zero_grad(set_to_none=True)
        loss = self.model.weight.square().sum() + epoch
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    def validation_loss(self, epoch: int) -> float:
        return float(3 - epoch)


def test_best_state_is_strictly_loadable() -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.training import fit_runtime

    runtime = _Runtime()
    optimizer = torch.optim.Adam(runtime.model.parameters(), lr=0.01)
    result = fit_runtime(runtime, optimizer, max_epochs=2, ema_decay=0.9)

    runtime.model.load_state_dict(result.best_raw, strict=True)
    assert set(result.best_ema) == set(runtime.model.state_dict())


def _config():
    return SimpleNamespace(
        bootstrap=SimpleNamespace(ensemble_size=0),
        prediction=SimpleNamespace(splits=("val", "test"), parameter_modes=("raw",)),
        data=SimpleNamespace(
            units=SimpleNamespace(
                energy="eV", forces="eV/Angstrom", stress="eV/Angstrom^3"
            )
        ),
    )


def test_converter_rejects_audit_inside_read_only_source(tmp_path: Path) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.errors import HardFailure
    from Uncertainty_Quantification.BootStrapping.internal_migration.migration.converter import (
        convert_legacy_run,
    )

    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(HardFailure, match="audit_root"):
        convert_legacy_run(
            source,
            tmp_path / "published",
            _config(),
            source / "audit",
        )


def test_member_store_uses_public_pt_names(tmp_path: Path) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.members import MemberStore

    store = MemberStore(tmp_path / "member_000")
    assert (store.best.name, store.final.name, store.latest.name) == (
        "best.pt",
        "final.pt",
        "latest.pt",
    )
