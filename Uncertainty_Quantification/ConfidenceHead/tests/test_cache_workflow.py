from __future__ import annotations

import json
import sys
import types
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from Uncertainty_Quantification.ConfidenceHead.confidence_head import (
    data as data_module,
)
from Uncertainty_Quantification.ConfidenceHead.confidence_head.config import (
    ConfidenceConfig,
)
from Uncertainty_Quantification.ConfidenceHead.confidence_head.data import (
    DatasetIdentity,
)


workflow = import_module(
    "Uncertainty_Quantification.ConfidenceHead.confidence_head.workflows.build_cache"
)


def _config(tmp_path: Path) -> ConfidenceConfig:
    return ConfidenceConfig.model_validate(
        {
            "profile": "production",
            "checkpoint": {
                "path": tmp_path / "model.ckpt",
                "expected_sha256": "a" * 64,
            },
            "data": {
                name: {
                    "path": tmp_path / f"{name}.extxyz",
                    "expected_sha256": str(index) * 64,
                }
                for index, name in enumerate(("train", "validation", "test"), start=1)
            },
            "run": {"output_root": tmp_path / "output"},
        }
    )


def test_execution_policy_is_float32_and_amp_requires_cuda() -> None:
    cpu = workflow._execution_policy(
        SimpleNamespace(run=SimpleNamespace(device="cpu", amp=False))
    )
    assert cpu == {
        "device": "cpu",
        "model_dtype": "float32",
        "system_dtype": "float32",
        "autocast": False,
        "autocast_dtype": None,
    }
    cuda = workflow._execution_policy(
        SimpleNamespace(run=SimpleNamespace(device="cuda:0", amp=True))
    )
    assert cuda["autocast"] is True
    assert cuda["autocast_dtype"] == "float16"
    with pytest.raises(ValueError, match="CUDA"):
        workflow._execution_policy(
            SimpleNamespace(run=SimpleNamespace(device="cpu", amp=True))
        )


def test_build_systems_adds_every_requested_neighbor_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    converted: list[tuple[Any, dict[str, Any]]] = []
    systems = [object(), object()]
    added: list[tuple[Any, Any, bool]] = []

    def systems_to_torch(atoms: Any, **kwargs: Any) -> Any:
        converted.append((atoms, kwargs))
        return systems[len(converted) - 1]

    class FakeNeighborList:
        def __init__(self, *, options: Any, **kwargs: Any) -> None:
            self.options = options
            assert kwargs == {
                "length_unit": "angstrom",
                "check_consistency": False,
            }

        def add_neighbor_list(self, system: Any, *, copy: bool) -> None:
            added.append((self.options, system, copy))

    systems_module = types.ModuleType("metatomic.torch.systems_to_torch")
    systems_module.systems_to_torch = systems_to_torch  # type: ignore[attr-defined]
    vesin_module = types.ModuleType("vesin.metatomic")
    vesin_module.NeighborList = FakeNeighborList  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "metatomic.torch.systems_to_torch", systems_module)
    monkeypatch.setitem(sys.modules, "vesin.metatomic", vesin_module)
    samples = [SimpleNamespace(atoms="a"), SimpleNamespace(atoms="b")]
    model = SimpleNamespace(requested_neighbor_lists=lambda: ["short", "long"])

    result = workflow._build_systems(samples, model, torch.device("cpu"))

    assert result == systems
    assert [item[0] for item in converted] == ["a", "b"]
    assert all(item[1]["dtype"] == torch.float32 for item in converted)
    assert all(item[1]["positions_requires_grad"] is False for item in converted)
    assert all(item[1]["cell_requires_grad"] is False for item in converted)
    assert added == [
        ("short", systems[0], True),
        ("long", systems[0], True),
        ("short", systems[1], True),
        ("long", systems[1], True),
    ]


def test_checkpoint_failure_leaves_an_incomplete_staging_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(
        workflow,
        "load_upet_checkpoint",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad checkpoint")),
    )

    with pytest.raises(ValueError, match="bad checkpoint"):
        workflow.build_cache(config)

    manifests = list(
        (config.run.output_root / "cache").glob(".staging-*/manifest.json")
    )
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["status"] == "incomplete"
    assert manifest["splits"] == {}


def test_identity_records_outputs_execution_versions_and_feature_dims(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    identities = {
        split: DatasetIdentity(Path(split), str(index) * 64, 1, 2, 6)
        for index, split in enumerate(("train", "validation", "test"), start=1)
    }
    monkeypatch.setattr(workflow, "_version", lambda name: f"{name}-version")
    monkeypatch.setattr(workflow, "_git_revision", lambda root: "git-revision")
    execution = workflow._execution_policy(config)
    outputs = {
        "energy_prediction": config.readouts.energy_prediction,
        "force_prediction": config.readouts.force_prediction,
        "energy_features": config.readouts.energy_features,
        "force_features": config.readouts.force_features,
    }

    payload = workflow._identity_payload(
        config, "c" * 64, identities, outputs, execution, (7, 11)
    )
    encoded = json.dumps(payload)

    assert payload["outputs"] == outputs
    assert payload["features"] == {
        "force_dim": 7,
        "energy_dim": 11,
        "dtype": "float32",
    }
    assert payload["execution"] == execution
    assert payload["cache"] == {"batch_size": config.cache.batch_size}
    assert set(payload["versions"]) == {
        "python",
        "torch",
        "metatomic",
        "metatrain",
        "upet",
        "upet_git",
    }
    assert str(config.run.output_root) not in encoded
    for forbidden in ("binning", "hidden_dims", "loss", "cumulant"):
        assert forbidden not in encoded


def test_expected_dataset_sha_is_rejected_before_parsing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "data.extxyz"
    path.write_bytes(b"wrong dataset")
    events: list[str] = []

    def fake_sha(candidate: Path) -> str:
        assert candidate == path
        events.append("sha")
        return "f" * 64

    def forbidden_parse(candidate: Path) -> Any:
        events.append("parse")
        raise AssertionError(f"must not parse {candidate}")

    monkeypatch.setattr(data_module, "sha256_file", fake_sha)
    monkeypatch.setattr(data_module, "iter_samples", forbidden_parse)

    with pytest.raises(ValueError, match="SHA mismatch"):
        data_module.dataset_identity(path, expected_sha256="e" * 64)

    assert events == ["sha"]


def test_extraction_rechecks_dataset_sha_when_stream_is_exhausted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "data.extxyz"
    monkeypatch.setattr(workflow, "iter_samples", lambda candidate: iter(()))
    monkeypatch.setattr(workflow, "sha256_file", lambda candidate: "f" * 64)
    config = SimpleNamespace(
        cache=SimpleNamespace(batch_size=2),
        readouts=SimpleNamespace(),
    )

    stream = workflow._raw_structures(
        path=path,
        model=object(),
        config=config,
        device=torch.device("cpu"),
        execution={"autocast": False},
        expected_sha256="e" * 64,
    )

    with pytest.raises(ValueError, match="changed during extraction"):
        list(stream)
