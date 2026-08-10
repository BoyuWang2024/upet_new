from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from Uncertainty_Quantification.FGE.internal_migration.migration import converter
from Uncertainty_Quantification.FGE.internal_migration.migration.converter import (
    _restart_materialized_state as materialize_restart,
)


def test_restart_materializer_consumes_open_authenticated_snapshot(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint = tmp_path / "base.ckpt"
    torch.save({"trusted": torch.tensor([7.0])}, checkpoint)
    trusted_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    replacement = tmp_path / "replacement.ckpt"
    torch.save({"trusted": torch.tensor([99.0])}, replacement)

    class FakeModel:
        def __init__(self, raw) -> None:
            self.raw = raw

        def to(self, **kwargs):
            assert kwargs == {"device": "cpu", "dtype": torch.float32}
            return self

        def state_dict(self):
            return {"frozen.weight": self.raw["trusted"]}

    def model_from_checkpoint(raw, *, context: str):
        assert context == "restart"
        return FakeModel(raw)

    monkeypatch.setattr(
        "metatrain.utils.io.model_from_checkpoint", model_from_checkpoint
    )
    real_fdopen = converter.os.fdopen

    def replacing_fdopen(descriptor: int, mode: str):
        handle = real_fdopen(descriptor, mode)
        replacement.replace(checkpoint)
        return handle

    monkeypatch.setattr(converter.os, "fdopen", replacing_fdopen)

    state = materialize_restart(checkpoint, trusted_sha)

    assert torch.equal(state["frozen.weight"], torch.tensor([7.0]))
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() != trusted_sha
