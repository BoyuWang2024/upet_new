from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from Uncertainty_Quantification.FGE.fge.errors import HardFailure
from Uncertainty_Quantification.FGE.internal_migration.migration import converter
from Uncertainty_Quantification.FGE.internal_migration.migration.converter import (
    _restart_materialized_state as materialize_restart,
)


def test_restart_materializer_rejects_wrong_sha_before_deserialization(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint = tmp_path / "member.ckpt"
    checkpoint.write_bytes(b"authenticated bytes")
    calls = {"torch_load": 0, "model_loader": 0}

    def forbidden_torch_load(*args, **kwargs):
        calls["torch_load"] += 1
        raise AssertionError("torch.load called before SHA validation")

    def forbidden_model_loader(*args, **kwargs):
        calls["model_loader"] += 1
        raise AssertionError("model loader called before SHA validation")

    monkeypatch.setattr(converter.torch, "load", forbidden_torch_load)
    monkeypatch.setattr(
        "metatrain.utils.io.model_from_checkpoint", forbidden_model_loader
    )

    with pytest.raises(HardFailure, match="SHA256"):
        materialize_restart(checkpoint, "0" * 64)

    assert calls == {"torch_load": 0, "model_loader": 0}


def test_restart_materializer_rejects_oversized_snapshot_before_deserialization(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint = tmp_path / "member.ckpt"
    checkpoint.write_bytes(b"too-large")
    monkeypatch.setattr(converter, "_MAX_CHECKPOINT_BYTES", 4)
    monkeypatch.setattr(
        converter.torch,
        "load",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("oversized checkpoint was deserialized")
        ),
    )

    with pytest.raises(HardFailure, match="size limit"):
        materialize_restart(
            checkpoint, hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        )


def test_restart_materializer_normalizes_schema_loader_errors(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint = tmp_path / "member.ckpt"
    checkpoint.write_bytes(b"serialized")
    monkeypatch.setattr(converter.torch, "load", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        "metatrain.utils.io.model_from_checkpoint",
        lambda *args, **kwargs: (_ for _ in ()).throw(KeyError("schema")),
    )

    with pytest.raises(HardFailure, match="materialize authenticated restart"):
        materialize_restart(
            checkpoint, hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        )
