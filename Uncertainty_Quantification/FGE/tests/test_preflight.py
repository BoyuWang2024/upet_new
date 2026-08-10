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
    assert set(report) == {
        "stage",
        "status",
        "basis",
        "identity",
        "scientific_flags",
        "config_identity",
    }
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
    """Canonical basis requires a complete training prior and no provenance."""
    from Uncertainty_Quantification.FGE.fge.preflight import run_preflight

    config = _config(tmp_path)
    run_preflight(config, "train")
    prior = _write_predict_prior(config)

    report = run_preflight(config, "predict", basis="canonical_artifacts")
    assert report["basis"] == "canonical_artifacts"

    training = json.loads(prior.read_text(encoding="utf-8"))
    training["migration"] = {"source": "old-result"}
    prior.write_text(json.dumps(training), encoding="utf-8")
    with pytest.raises(HardFailure):
        run_preflight(config, "predict", basis="canonical_artifacts")


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write_predict_prior(config: Any) -> Path:
    """Create a complete training prior but deliberately no prediction manifest."""
    from Uncertainty_Quantification.FGE.tests.test_validation import _formal_a3_payload

    root = config.paths.output_root / config.project.name
    resolved = config.sanitized()
    resolved["ema"] = {"member_source": "raw_endpoint"}
    config.sanitized = lambda: dict(resolved)
    (root / "config_resolved.yaml").write_text(json.dumps(resolved), encoding="utf-8")
    members: list[dict[str, object]] = []
    for index in range(1, 3):
        path = root / "training" / "members" / f"member_{index:03d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload, _ = _formal_a3_payload(index, config.identity.base_checkpoint_sha256)
        torch.save(payload, path)
        members.append(
            {
                "member_id": f"member_{index:03d}",
                "sha256": _sha256(path),
                "cycle": index,
                "endpoint_global_step": index,
            }
        )
    training = {
        "schema_version": "upet.fge.training.v1",
        "project_name": config.project.name,
        "config_resolved": resolved,
        "config_identity": {"sha256": _canonical_hash(resolved)},
        "checkpoint_identity": {
            "sha256": config.identity.base_checkpoint_sha256,
        },
        "data_identities": {
            "train": {"sha256": config.identity.train_data_sha256},
            "val": {"sha256": config.identity.val_data_sha256},
            "test": {"sha256": config.identity.test_data_sha256},
        },
        "model_contract": {
            "readout_tensor_count": 12,
            "readout_parameter_count": 13338,
        },
        "frozen_fingerprint_identity": {"sha256": "f" * 64},
        "dependency_snapshot": {"torch": "2.x", "metatrain": "2026.3.1"},
        "scientific_flags": {
            "path_feasibility_only": True,
            "split_leakage": True,
            "scientific_evaluation": False,
            "inference_only": False,
        },
        "training_code_identity": {"status": "unavailable"},
        "artifact_writer_code_identity": {
            "commit": "a" * 40,
            "dirty_sha256": "a" * 64,
        },
        "validator_code_identity": {
            "commit": "b" * 40,
            "dirty_sha256": "b" * 64,
        },
        "member_count": 2,
        "members": members,
    }
    path = root / "training" / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(training), encoding="utf-8")
    return path


def test_predict_canonical_preflight_only_requires_verified_training_prior(
    tmp_path: Path,
) -> None:
    """Predict starts after training; prediction artifacts do not exist yet."""
    from Uncertainty_Quantification.FGE.fge.preflight import run_preflight

    config = _config(tmp_path)
    run_preflight(config, "train")
    prior = _write_predict_prior(config)

    report = run_preflight(config, "predict", basis="canonical_artifacts")

    assert report["stage"] == "predict"
    assert prior.is_file()
    assert not (prior.parents[1] / "prediction" / "manifest.json").exists()


def test_predict_canonical_preflight_rejects_training_identity_or_provenance_drift(
    tmp_path: Path,
) -> None:
    """Prior config/hash/schema and explicit migration provenance are strict."""
    from Uncertainty_Quantification.FGE.fge.preflight import run_preflight

    config = _config(tmp_path)
    run_preflight(config, "train")
    prior = _write_predict_prior(config)
    training = json.loads(prior.read_text(encoding="utf-8"))
    training["config_identity"] = {"sha256": "0" * 64}
    prior.write_text(json.dumps(training), encoding="utf-8")

    with pytest.raises(HardFailure):
        run_preflight(config, "predict", basis="canonical_artifacts")

    training["config_identity"] = {"sha256": _canonical_hash(config.sanitized())}
    training["migration"] = {"source": "old"}
    prior.write_text(json.dumps(training), encoding="utf-8")
    with pytest.raises(HardFailure):
        run_preflight(config, "predict", basis="canonical_artifacts")


def test_evaluate_canonical_preflight_reopens_payload_and_checks_metadata(
    tmp_path: Path,
) -> None:
    """Evaluate cannot trust a self-consistent hash for invalid prediction metadata."""
    from Uncertainty_Quantification.FGE.fge.preflight import run_preflight
    from Uncertainty_Quantification.FGE.tests.test_validation import _payload

    config = _config(tmp_path)
    run_preflight(config, "train")
    _write_predict_prior(config)
    root = config.paths.output_root / config.project.name
    payload = _payload(2)
    prediction_path = root / "prediction" / "test_raw.pt"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, prediction_path)
    manifest_path = root / "prediction" / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "upet.fge.prediction.v1",
                "member_ids": ["member_001", "member_002"],
                "shape": {"K": 2, "S": 1, "A": 1},
                "artifact": {
                    "role": "prediction",
                    "path": "prediction/test_raw.pt",
                    "bytes": prediction_path.stat().st_size,
                    "sha256": _sha256(prediction_path),
                },
            }
        ),
        encoding="utf-8",
    )
    run_preflight(config, "evaluate", basis="canonical_artifacts")

    payload["target_names"]["energy"] = "wrong"
    torch.save(payload, prediction_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = manifest["artifact"]
    assert isinstance(artifact, dict)
    artifact["bytes"] = prediction_path.stat().st_size
    artifact["sha256"] = _sha256(prediction_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(HardFailure):
        run_preflight(config, "evaluate", basis="canonical_artifacts")


def test_canonical_preflight_binds_disk_resolved_configuration(
    tmp_path: Path,
) -> None:
    """Prior manifest cannot substitute for the canonical config file on disk."""
    from Uncertainty_Quantification.FGE.fge.preflight import run_preflight

    config = _config(tmp_path)
    run_preflight(config, "train")
    _write_predict_prior(config)
    root = config.paths.output_root / config.project.name
    resolved = root / "config_resolved.yaml"
    resolved.unlink()

    with pytest.raises(HardFailure):
        run_preflight(config, "predict", basis="canonical_artifacts")

    resolved.write_text(json.dumps({"paths": {}}), encoding="utf-8")
    with pytest.raises(HardFailure):
        run_preflight(config, "predict", basis="canonical_artifacts")
