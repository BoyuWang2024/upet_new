"""Disk-backed result-validation contracts."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from Uncertainty_Quantification.FGE.fge.evaluation import evaluate_prediction
from Uncertainty_Quantification.FGE.fge.members import ReadoutAudit


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _identity(letter: str) -> dict[str, str]:
    return {"sha256": letter * 64}


def _config_identity(config_resolved: Mapping[str, object]) -> dict[str, str]:
    encoded = json.dumps(config_resolved, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return {"sha256": hashlib.sha256(encoded).hexdigest()}


def _code_identity(letter: str) -> dict[str, str]:
    return {"commit": letter * 40, "dirty_sha256": letter * 64}


def _payload(member_count: int) -> dict[str, Any]:
    first = torch.zeros((1,), dtype=torch.float32)
    second = torch.full((1,), 2.0, dtype=torch.float32)
    energies = torch.stack(
        [first if index % 2 == 0 else second for index in range(member_count)]
    )
    force0 = torch.zeros((1, 3), dtype=torch.float32)
    force2 = torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float32)
    forces = torch.stack(
        [force0 if index % 2 == 0 else force2 for index in range(member_count)]
    )
    stress = torch.zeros((member_count, 1, 3, 3), dtype=torch.float32)
    member_ids = tuple(f"member_{index:03d}" for index in range(1, member_count + 1))
    return {
        "energy_prediction": energies,
        "forces_prediction": forces,
        "stress_prediction": stress,
        "energy_reference": torch.ones(1, dtype=torch.float32),
        "forces_reference": torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
        "stress_reference": torch.zeros((1, 3, 3), dtype=torch.float32),
        "n_atoms": torch.ones(1, dtype=torch.int64),
        "structure_offsets": torch.tensor([0, 1], dtype=torch.int64),
        "member_ids": member_ids,
        "structure_ids": ("structure_000",),
        "atomic_numbers": torch.tensor([1], dtype=torch.int64),
        "structure_mapping": torch.tensor([0], dtype=torch.int64),
        "target_names": {
            "energy": "energy",
            "forces": "non_conservative_forces",
            "stress": "non_conservative_stress",
        },
        "units": {
            "energy": "eV",
            "forces": "eV/angstrom",
            "stress": "eV/angstrom^3",
        },
        "statistics": {"K": member_count, "S": 1, "A": 1},
    }


def _config(
    project_name: str, member_count: int, config_resolved: Mapping[str, object]
) -> Any:
    config = SimpleNamespace(
        project=SimpleNamespace(name=project_name),
        fge=SimpleNamespace(member_count=member_count),
        identity=SimpleNamespace(
            base_checkpoint_sha256="a" * 64,
            train_data_sha256="b" * 64,
            val_data_sha256="c" * 64,
            test_data_sha256="d" * 64,
        ),
        data=SimpleNamespace(
            energy_target="energy",
            forces_target="non_conservative_forces",
            stress_target="non_conservative_stress",
            energy_unit="eV",
            forces_unit="eV/angstrom",
            stress_unit="eV/angstrom^3",
        ),
        training=SimpleNamespace(device="cpu", dtype="float32"),
        scientific=SimpleNamespace(
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
        ),
        evaluation=SimpleNamespace(
            formula_version="legacy_upet_fge_v1",
            metric_schema_version=4,
            risk_coverages=(1.0, 0.95, 0.9, 0.8, 0.7, 0.5, 0.3, 0.1),
            constant_tolerance=1e-12,
        ),
    )
    config.sanitized = lambda: dict(config_resolved)
    return config


def _formal_a3_payload(
    index: int, base_sha256: str
) -> tuple[dict[str, object], ReadoutAudit]:
    """Return a genuine 12-tensor / 13,338-scalar A3 wire payload."""
    names = tuple(f"node_last_layers.layer_{number:02d}" for number in range(12))
    shapes = tuple((1111,) for _ in range(11)) + ((1117,),)
    values = tuple(torch.zeros(shape, dtype=torch.float32) for shape in shapes)
    audit = ReadoutAudit(
        names=names,
        tensor_count=12,
        scalar_count=13338,
        shapes=shapes,
        dtypes=("torch.float32",) * 12,
        all_finite=True,
    )
    return (
        {
            "schema_version": "upet.fge.member-delta.v1",
            "member_id": index,
            "cycle": index,
            "global_step": index,
            "base_sha256": base_sha256,
            "tensors": [
                {
                    "name": name,
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                    "value": value,
                }
                for name, value in zip(names, values, strict=True)
            ],
        },
        audit,
    )


def make_canonical_result(
    tmp_path: Path, *, project_name: str = "upet_fge_n20_cpu", member_count: int = 2
) -> tuple[Any, Path]:
    """Write a complete tiny formal tree from literals, not production builders."""
    root = tmp_path / project_name
    root.mkdir(parents=True)
    path_hashes = {
        "base_checkpoint": "a" * 64,
        "train_data": "b" * 64,
        "val_data": "c" * 64,
        "test_data": "d" * 64,
    }
    config_resolved = {
        "paths": {
            role: {"role": role, "sha256": digest}
            for role, digest in path_hashes.items()
        },
        "ema": {"member_source": "raw_endpoint"},
    }
    (root / "config_resolved.yaml").write_text(
        json.dumps(config_resolved), encoding="utf-8"
    )
    flags = {
        "path_feasibility_only": True,
        "split_leakage": True,
        "scientific_evaluation": False,
        "inference_only": False,
    }
    for stage in ("train", "predict", "evaluate"):
        _write_json(
            root / "preflight" / f"{stage}.json",
            {
                "stage": stage,
                "status": "PASS",
                "basis": "runtime_inputs",
                "identity": {
                    "inputs": path_hashes,
                    "targets": {
                        "energy": "energy",
                        "forces": "non_conservative_forces",
                        "stress": "non_conservative_stress",
                    },
                    "units": {
                        "energy": "eV",
                        "forces": "eV/angstrom",
                        "stress": "eV/angstrom^3",
                    },
                    "model_contract": {
                        "readout_tensor_count": 12,
                        "readout_parameter_count": 13338,
                    },
                    "member_count": member_count,
                    "runtime": {"device": "cpu", "dtype": "float32"},
                },
                "scientific_flags": (
                    flags
                    if stage == "train"
                    else {
                        **flags,
                        "inference_only": True,
                    }
                ),
                "config_identity": _config_identity(config_resolved),
            },
        )
    members: list[dict[str, object]] = []
    for index in range(1, member_count + 1):
        member_path = root / "training" / "members" / f"member_{index:03d}.pt"
        member_path.parent.mkdir(parents=True, exist_ok=True)
        member_payload, _ = _formal_a3_payload(index, path_hashes["base_checkpoint"])
        torch.save(member_payload, member_path)
        members.append(
            {
                "member_id": f"member_{index:03d}",
                "sha256": _sha256(member_path),
                "cycle": index,
                "endpoint_global_step": index,
            }
        )
    training = {
        "schema_version": "upet.fge.training.v1",
        "project_name": project_name,
        "config_resolved": config_resolved,
        "config_identity": _config_identity(config_resolved),
        "checkpoint_identity": _identity("a"),
        "data_identities": {
            "train": _identity("b"),
            "val": _identity("c"),
            "test": _identity("d"),
        },
        "model_contract": {
            "readout_tensor_count": 12,
            "readout_parameter_count": 13338,
        },
        "frozen_fingerprint_identity": _identity("f"),
        "dependency_snapshot": {"torch": "2.x", "metatrain": "2026.3.1"},
        "scientific_flags": flags,
        "training_code_identity": {"status": "unavailable"},
        "artifact_writer_code_identity": _code_identity("a"),
        "validator_code_identity": _code_identity("b"),
        "member_count": member_count,
        "members": members,
    }
    _write_json(root / "training" / "manifest.json", training)
    payload = _payload(member_count)
    prediction_path = root / "prediction" / "test_raw.pt"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, prediction_path)
    _write_json(
        root / "prediction" / "manifest.json",
        {
            "schema_version": "upet.fge.prediction.v1",
            "member_ids": list(payload["member_ids"]),
            "shape": {"K": member_count, "S": 1, "A": 1},
            "artifact": {
                "role": "prediction",
                "path": "prediction/test_raw.pt",
                "bytes": prediction_path.stat().st_size,
                "sha256": _sha256(prediction_path),
            },
            "artifact_writer_code_identity": _code_identity("a"),
            "validator_code_identity": _code_identity("b"),
        },
    )
    evaluation = root / "evaluation" / "legacy_equal_weight"
    evaluation.mkdir(parents=True)
    recomputed = evaluate_prediction(
        payload,
        (1.0, 0.95, 0.9, 0.8, 0.7, 0.5, 0.3, 0.1),
        1e-12,
    )
    torch.save(dict(recomputed.ensemble), evaluation / "ensemble.pt")
    torch.save(dict(recomputed.uncertainty), evaluation / "uncertainty.pt")
    _write_json(evaluation / "metrics.json", recomputed.metrics)
    (evaluation / "report.md").write_text(
        "# Canonical FGE report\n"
        + json.dumps(recomputed.report_inputs, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return _config(project_name, member_count, config_resolved), root


def test_validation_reopens_disk_artifacts_and_publishes_manifest_last(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Validation recomputes canonical quantities and writes completion last."""
    import Uncertainty_Quantification.FGE.fge.validation as validation

    config, root = make_canonical_result(tmp_path)
    writes: list[Path] = []
    original = validation.atomic_write_json

    def observed(path: str | Path, payload: Mapping[str, Any] | list[Any]) -> None:
        writes.append(Path(path))
        original(path, payload)

    monkeypatch.setattr(validation, "atomic_write_json", observed)
    report = validation.validate_result(config, root)

    assert report.status == "PASS"
    assert (root / "validation.json").is_file()
    assert (root / "result_manifest.json").is_file()
    assert writes[-2:] == [root / "validation.json", root / "result_manifest.json"]
    manifest = json.loads((root / "result_manifest.json").read_text(encoding="utf-8"))
    assert {entry["role"] for entry in manifest["artifacts"]} >= {
        "validation",
        "prediction",
        "ensemble",
        "uncertainty",
        "metrics",
    }
    assert all(
        entry["path"] != "result_manifest.json" for entry in manifest["artifacts"]
    )


def _training(root: Path) -> dict[str, object]:
    return json.loads((root / "training" / "manifest.json").read_text(encoding="utf-8"))


def _write_training(root: Path, payload: Mapping[str, object]) -> None:
    _write_json(root / "training" / "manifest.json", payload)


def _member_path(root: Path, index: int = 1) -> Path:
    return root / "training" / "members" / f"member_{index:03d}.pt"


def _refresh_member_hash(root: Path, index: int = 1) -> None:
    training = _training(root)
    members = training["members"]
    assert isinstance(members, list)
    member = members[index - 1]
    assert isinstance(member, dict)
    member["sha256"] = _sha256(_member_path(root, index))
    _write_training(root, training)


def test_validation_accepts_approved_ema_member_source_but_rejects_provenance(
    tmp_path: Path,
) -> None:
    """The formal config field is not confused with migration provenance."""
    from Uncertainty_Quantification.FGE.fge.errors import HardFailure
    from Uncertainty_Quantification.FGE.fge.validation import validate_result

    config, root = make_canonical_result(tmp_path)
    assert validate_result(config, root, publish_completion=False).status == "PASS"

    training = _training(root)
    resolved = training["config_resolved"]
    assert isinstance(resolved, dict)
    resolved["migration"] = {"source": "old-result"}
    _write_training(root, training)

    with pytest.raises(HardFailure):
        validate_result(config, root, publish_completion=False)


@pytest.mark.parametrize("mutation", ["base", "name", "order", "dtype", "shape"])
def test_validation_rejects_each_a3_contract_drift(
    tmp_path: Path, mutation: str
) -> None:
    """Validation reopens actual A3 payloads, not just their outer hashes."""
    from Uncertainty_Quantification.FGE.fge.errors import HardFailure
    from Uncertainty_Quantification.FGE.fge.validation import validate_result

    config, root = make_canonical_result(tmp_path)
    path = _member_path(root)
    payload = torch.load(path, weights_only=True)
    assert isinstance(payload, dict)
    tensors = payload["tensors"]
    assert isinstance(tensors, list)
    if mutation == "base":
        payload["base_sha256"] = "f" * 64
    elif mutation == "name":
        assert isinstance(tensors[0], dict)
        tensors[0]["name"] = "node_last_layers.tampered"
    elif mutation == "order":
        tensors.reverse()
    elif mutation == "dtype":
        assert isinstance(tensors[0], dict)
        tensors[0]["dtype"] = "torch.float64"
    else:
        assert isinstance(tensors[0], dict)
        tensors[0]["shape"] = [1110]
    torch.save(payload, path)
    _refresh_member_hash(root)

    with pytest.raises(HardFailure):
        validate_result(config, root, publish_completion=False)


def test_validation_binds_disk_resolved_config_and_config_identity_to_argument(
    tmp_path: Path,
) -> None:
    """A different caller configuration cannot validate an unrelated result."""
    from Uncertainty_Quantification.FGE.fge.errors import HardFailure
    from Uncertainty_Quantification.FGE.fge.validation import validate_result

    config, root = make_canonical_result(tmp_path)
    changed = dict(config.sanitized())
    changed["ema"] = {"member_source": "not_raw_endpoint"}
    config.sanitized = lambda: changed

    with pytest.raises(HardFailure):
        validate_result(config, root, publish_completion=False)


def test_validation_recomputes_all_durable_uq_metrics_and_report_inputs(
    tmp_path: Path,
) -> None:
    """Changing an unchecked force UQ/metric/report input must hard-fail."""
    from Uncertainty_Quantification.FGE.fge.errors import HardFailure
    from Uncertainty_Quantification.FGE.fge.validation import validate_result

    config, root = make_canonical_result(tmp_path)
    evaluation = root / "evaluation" / "legacy_equal_weight"
    uncertainty = torch.load(evaluation / "uncertainty.pt", weights_only=True)
    assert isinstance(uncertainty, dict)
    force_component = uncertainty["force_component"]
    assert isinstance(force_component, dict)
    force_component["std"] = torch.full((1, 3), 7.0, dtype=torch.float32)
    torch.save(uncertainty, evaluation / "uncertainty.pt")

    with pytest.raises(HardFailure):
        validate_result(config, root, publish_completion=False)


def test_completed_manifest_requires_exact_canonical_role_inventory_and_safe_paths(
    tmp_path: Path,
) -> None:
    """Completion cannot reference outside files or arbitrary role aliases."""
    from Uncertainty_Quantification.FGE.fge.errors import HardFailure
    from Uncertainty_Quantification.FGE.fge.validation import validate_result

    config, root = make_canonical_result(tmp_path)
    validate_result(config, root)
    manifest_path = root / "result_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest["artifacts"]
    assert isinstance(artifacts, list)
    first = artifacts[0]
    assert isinstance(first, dict)
    outside = root.parent / "outside.bin"
    outside.write_bytes(b"outside")
    first["path"] = "../outside.bin"
    first["sha256"] = _sha256(outside)
    first["bytes"] = outside.stat().st_size
    _write_json(manifest_path, manifest)

    with pytest.raises(HardFailure):
        validate_result(config, root)


def test_completed_manifest_rejects_noncanonical_role_even_with_valid_hash(
    tmp_path: Path,
) -> None:
    """A manifest role is an exact canonical reference, not free-form text."""
    from Uncertainty_Quantification.FGE.fge.errors import HardFailure
    from Uncertainty_Quantification.FGE.fge.validation import validate_result

    config, root = make_canonical_result(tmp_path)
    validate_result(config, root)
    manifest_path = root / "result_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest["artifacts"]
    assert isinstance(artifacts, list)
    first = artifacts[0]
    assert isinstance(first, dict)
    first["role"] = "anything"
    _write_json(manifest_path, manifest)

    with pytest.raises(HardFailure):
        validate_result(config, root)


@pytest.mark.parametrize(
    "relative",
    [
        "training/members/member_003.pt",
        "wandb/run.json",
        "logs/train.log",
        "plots/summary.png",
        "_work/run_state.pt",
    ],
)
def test_first_validation_rejects_unallowed_residue_before_publication(
    tmp_path: Path, relative: str
) -> None:
    """First publication scans the formal tree instead of inventorying a subset."""
    from Uncertainty_Quantification.FGE.fge.errors import HardFailure
    from Uncertainty_Quantification.FGE.fge.validation import validate_result

    config, root = make_canonical_result(tmp_path)
    residue = root / relative
    residue.parent.mkdir(parents=True, exist_ok=True)
    if residue.suffix == ".pt":
        residue.write_bytes(_member_path(root).read_bytes())
    else:
        residue.write_text("residue\n", encoding="utf-8")

    with pytest.raises(HardFailure):
        validate_result(config, root, publish_completion=False)


def test_validation_rejects_preflight_report_identity_or_config_drift(
    tmp_path: Path,
) -> None:
    """Every durable preflight report binds its full formal identity."""
    from Uncertainty_Quantification.FGE.fge.errors import HardFailure
    from Uncertainty_Quantification.FGE.fge.validation import validate_result

    config, root = make_canonical_result(tmp_path)
    report_path = root / "preflight" / "evaluate.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["identity"] = {"member_count": 99}
    _write_json(report_path, report)

    with pytest.raises(HardFailure):
        validate_result(config, root, publish_completion=False)


@pytest.mark.parametrize("kind", ["empty_logs", "wandb_symlink"])
def test_first_validation_rejects_unallowed_directory_residue(
    tmp_path: Path, kind: str
) -> None:
    """Directories are part of the formal tree and cannot hide residue."""
    from Uncertainty_Quantification.FGE.fge.errors import HardFailure
    from Uncertainty_Quantification.FGE.fge.validation import validate_result

    config, root = make_canonical_result(tmp_path)
    if kind == "empty_logs":
        (root / "logs").mkdir()
    else:
        outside = tmp_path / "outside_wandb"
        outside.mkdir()
        os.symlink(outside, root / "wandb", target_is_directory=True)

    with pytest.raises(HardFailure):
        validate_result(config, root, publish_completion=False)


def test_validation_binds_training_scientific_flags_to_configuration(
    tmp_path: Path,
) -> None:
    """Training flags in manifest are formal scientific identity, not decoration."""
    from Uncertainty_Quantification.FGE.fge.errors import HardFailure
    from Uncertainty_Quantification.FGE.fge.validation import validate_result

    config, root = make_canonical_result(tmp_path)
    training = _training(root)
    flags = training["scientific_flags"]
    assert isinstance(flags, dict)
    flags["split_leakage"] = False
    _write_training(root, training)

    with pytest.raises(HardFailure):
        validate_result(config, root, publish_completion=False)


def test_validation_rejects_numeric_training_scientific_flags(
    tmp_path: Path,
) -> None:
    """A JSON 0 cannot impersonate the boolean False scientific flag."""
    from Uncertainty_Quantification.FGE.fge.errors import HardFailure
    from Uncertainty_Quantification.FGE.fge.validation import validate_result

    config, root = make_canonical_result(tmp_path)
    training = _training(root)
    flags = training["scientific_flags"]
    assert isinstance(flags, dict)
    flags["scientific_evaluation"] = 0
    _write_training(root, training)

    with pytest.raises(HardFailure):
        validate_result(config, root, publish_completion=False)
