"""Contract tests for compute-free FGE stage preflight."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from Uncertainty_Quantification.FGE.fge.errors import HardFailure


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config(tmp_path: Path) -> Any:
    """Build a small runtime basis without using any production builders."""
    base = tmp_path / "base.ckpt"
    torch.save({"model_state_dict": {}}, base)
    train = tmp_path / "train.extxyz"
    val = tmp_path / "val.extxyz"
    test = tmp_path / "test.extxyz"
    for path, content in ((train, b"train"), (val, b"val"), (test, b"test")):
        path.write_bytes(content)
    output_root = tmp_path / "outputs"
    identity = SimpleNamespace(
        base_checkpoint_sha256=_sha256(base),
        train_data_sha256=_sha256(train),
        val_data_sha256=_sha256(val),
        test_data_sha256=_sha256(test),
    )
    paths = SimpleNamespace(
        base_checkpoint=base,
        train_data=train,
        val_data=val,
        test_data=test,
        output_root=output_root,
    )
    data = SimpleNamespace(
        energy_target="energy",
        forces_target="non_conservative_forces",
        stress_target="non_conservative_stress",
        energy_unit="eV",
        forces_unit="eV/angstrom",
        stress_unit="eV/angstrom^3",
    )
    training = SimpleNamespace(
        expected_readout_tensor_count=12,
        expected_readout_parameter_count=13338,
        device="cpu",
        dtype="float32",
    )
    fge = SimpleNamespace(member_count=2)
    scientific = SimpleNamespace(
        training=SimpleNamespace(
            path_feasibility_only=True,
            split_leakage=True,
            scientific_evaluation=False,
            inference_only=False,
        ),
        evaluation=SimpleNamespace(
            path_feasibility_only=True,
            split_leakage=True,
            scientific_evaluation=False,
            inference_only=True,
        ),
    )
    config = SimpleNamespace(
        project=SimpleNamespace(name="upet_fge_n20_cpu"),
        paths=paths,
        identity=identity,
        data=data,
        training=training,
        fge=fge,
        scientific=scientific,
    )

    def sanitized() -> dict[str, object]:
        return {
            "paths": {
                "base_checkpoint": {
                    "role": "base_checkpoint",
                    "sha256": identity.base_checkpoint_sha256,
                },
                "train_data": {
                    "role": "train_data",
                    "sha256": identity.train_data_sha256,
                },
                "val_data": {"role": "val_data", "sha256": identity.val_data_sha256},
                "test_data": {"role": "test_data", "sha256": identity.test_data_sha256},
                "output_root": {"role": "output_root"},
            }
        }

    config.sanitized = sanitized
    return config


def _report_paths(config: Any) -> tuple[Path, Path]:
    root = config.paths.output_root / config.project.name  # type: ignore[attr-defined]
    return root / "config_resolved.yaml", root / "preflight" / "train.json"


@pytest.mark.parametrize("stage", ["", "validate", "forward", "train "])
def test_preflight_rejects_every_stage_outside_the_three_formal_stages(
    tmp_path: Path, stage: str
) -> None:
    """Only the formal train, predict, and evaluate stages can preflight."""
    from Uncertainty_Quantification.FGE.fge.preflight import run_preflight

    with pytest.raises(HardFailure):
        run_preflight(_config(tmp_path), stage)


def test_native_train_preflight_checks_runtime_basis_without_compute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The runtime gate validates identities and contracts without model compute."""
    from Uncertainty_Quantification.FGE.fge.preflight import run_preflight

    config = _config(tmp_path)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise AssertionError("preflight must not execute model compute")

    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(torch.optim.Optimizer, "step", forbidden)
    monkeypatch.setattr(torch.nn.Module, "_call_impl", forbidden)

    report = run_preflight(config, "train")
    resolved, train_report = _report_paths(config)

    assert report["stage"] == "train"
    assert report["status"] == "PASS"
    assert report["basis"] == "runtime_inputs"
    assert set(report) == {"stage", "status", "basis", "identity", "scientific_flags"}
    assert report["identity"]["model_contract"] == {
        "readout_tensor_count": 12,
        "readout_parameter_count": 13338,
    }
    assert report["identity"]["member_count"] == 2
    assert report["identity"]["runtime"] == {"device": "cpu", "dtype": "float32"}
    assert report["scientific_flags"] == {
        "path_feasibility_only": True,
        "split_leakage": True,
        "scientific_evaluation": False,
        "inference_only": False,
    }
    assert resolved.is_file()
    assert train_report.is_file()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda config: setattr(config.identity, "test_data_sha256", "0" * 64),
        lambda config: setattr(config.data, "energy_unit", "kcal/mol"),
        lambda config: setattr(config.training, "expected_readout_tensor_count", 11),
        lambda config: setattr(
            config.training, "expected_readout_parameter_count", 13337
        ),
        lambda config: setattr(config.fge, "member_count", 1),
        lambda config: setattr(config.training, "dtype", "float64"),
        lambda config: setattr(config.scientific.training, "split_leakage", False),
    ],
)
def test_runtime_preflight_rejects_each_changed_formal_contract(
    tmp_path: Path, mutate: Any
) -> None:
    """No hash, target/unit, scope, scale, runtime, or scientific drift is accepted."""
    from Uncertainty_Quantification.FGE.fge.preflight import run_preflight

    config = _config(tmp_path)
    mutate(config)

    with pytest.raises(HardFailure):
        run_preflight(config, "train")


def test_later_preflights_are_same_schema_and_do_not_rewrite_completed_results(
    tmp_path: Path,
) -> None:
    """Predict/evaluate only check the already-recorded native runtime identity."""
    from Uncertainty_Quantification.FGE.fge.preflight import run_preflight

    config = _config(tmp_path)
    train = run_preflight(config, "train")
    root = config.paths.output_root / config.project.name
    train_report = root / "preflight" / "train.json"
    before = train_report.read_bytes()
    before_mtime = train_report.stat().st_mtime_ns

    predict = run_preflight(config, "predict")
    evaluate = run_preflight(config, "evaluate")

    assert set(predict) == set(train) == set(evaluate)
    assert train_report.read_bytes() == before
    assert train_report.stat().st_mtime_ns == before_mtime
    assert (root / "preflight" / "predict.json").is_file()
    assert (root / "preflight" / "evaluate.json").is_file()


def test_canonical_preflight_reads_only_canonical_artifacts_and_rejects_provenance(
    tmp_path: Path,
) -> None:
    """Canonical basis is source-independent and checks files already on disk."""
    from Uncertainty_Quantification.FGE.fge.preflight import run_preflight

    config = _config(tmp_path)
    run_preflight(config, "train")
    root = config.paths.output_root / config.project.name
    (root / "training").mkdir(parents=True)
    (root / "prediction").mkdir()
    (root / "training" / "manifest.json").write_text(
        json.dumps({"schema_version": "upet.fge.training.v1", "member_count": 2}),
        encoding="utf-8",
    )
    (root / "prediction" / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "upet.fge.prediction.v1",
                "member_ids": ["member_001", "member_002"],
            }
        ),
        encoding="utf-8",
    )

    report = run_preflight(config, "predict", basis="canonical_artifacts")
    assert report["basis"] == "canonical_artifacts"

    (root / "training" / "manifest.json").write_text(
        json.dumps({"schema_version": "upet.fge.training.v1", "migration": True}),
        encoding="utf-8",
    )
    with pytest.raises(HardFailure):
        run_preflight(config, "predict", basis="canonical_artifacts")
