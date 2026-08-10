"""Disk-backed result-validation contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _identity(letter: str) -> dict[str, str]:
    return {"sha256": letter * 64}


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


def _config(project_name: str, member_count: int) -> Any:
    return SimpleNamespace(
        project=SimpleNamespace(name=project_name),
        fge=SimpleNamespace(member_count=member_count),
        evaluation=SimpleNamespace(
            formula_version="legacy_upet_fge_v1",
            metric_schema_version=4,
            risk_coverages=(1.0, 0.95, 0.9, 0.8, 0.7, 0.5, 0.3, 0.1),
            constant_tolerance=1e-12,
        ),
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
        }
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
                "identity": {"member_count": member_count},
                "scientific_flags": flags,
            },
        )
    members: list[dict[str, object]] = []
    for index in range(1, member_count + 1):
        member_path = root / "training" / "members" / f"member_{index:03d}.pt"
        member_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": "upet.fge.member-delta.v1",
                "member_id": f"member_{index:03d}",
                "base_sha256": path_hashes["base_checkpoint"],
                "tensors": {},
            },
            member_path,
        )
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
        "config_identity": _identity("e"),
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
    torch.save(
        {
            "energy": torch.ones(1, dtype=torch.float32),
            "forces": torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
            "stress": torch.zeros((1, 3, 3), dtype=torch.float32),
        },
        evaluation / "ensemble.pt",
    )
    torch.save(
        {
            "formula_version": "legacy_upet_fge_v1",
            "energy_total": {"std": torch.ones(1), "gmd": torch.ones(1)},
        },
        evaluation / "uncertainty.pt",
    )
    _write_json(
        evaluation / "metrics.json",
        {
            "schema_version": 4,
            "mae": {
                "energy_total": 0.0,
                "energy_per_atom": 0.0,
                "force_component": 0.0,
                "stress_component": 0.0,
            },
        },
    )
    (evaluation / "report.md").write_text("# Canonical FGE report\n", encoding="utf-8")
    return _config(project_name, member_count), root


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
