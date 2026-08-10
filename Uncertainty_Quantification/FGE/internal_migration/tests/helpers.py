from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import torch
import yaml

from Uncertainty_Quantification.FGE.fge.config import load_config
from Uncertainty_Quantification.FGE.fge.evaluation import evaluate_prediction
from Uncertainty_Quantification.FGE.internal_migration.migration.legacy_reader import (
    LegacyExpectations,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def build_conversion_case(
    legacy_tree, config_payload: dict[str, object], tmp_path: Path
):
    root, _, _, _ = legacy_tree
    names = [f"node_last_layers.tensor_{index:02d}" for index in range(11)]
    names.append("edge_last_layers.tensor_11")
    shapes = [(1,)] * 11 + [(13_327,)]

    def state(readout_value: float) -> dict[str, torch.Tensor]:
        result = {
            name: torch.full(shape, readout_value, dtype=torch.float32)
            for name, shape in zip(names, shapes, strict=False)
        }
        result["frozen.weight"] = torch.tensor([3.0], dtype=torch.float32)
        return result

    base = tmp_path / "base.ckpt"
    torch.save({"model_state_dict": state(0.0)}, base)
    member_hashes: dict[str, str] = {}
    manifest_path = root / "matpes_test_full/fge_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for index, entry in enumerate(manifest["members"], start=1):
        member_id = f"member_{index:03d}"
        path = root / "successful/members" / f"{member_id}.ckpt"
        torch.save({"model_state_dict": state(float(index))}, path)
        member_hashes[member_id] = _sha(path)
        entry["checkpoint_path"] = str(path.resolve())
    _write_json(manifest_path, manifest)

    prediction_root = root / "matpes_test_full/predictions/test"
    summary = json.loads((prediction_root / "prediction_summary.json").read_text())
    chunks = [
        [
            torch.load(
                prediction_root / relative, map_location="cpu", weights_only=True
            )
            for relative in summary["member_chunks"][member_id]
        ]
        for member_id in summary["member_ids"]
    ]
    payload = {
        "energy_prediction": torch.stack(
            [torch.cat([chunk["E_member"] for chunk in member]) for member in chunks]
        ),
        "forces_prediction": torch.stack(
            [torch.cat([chunk["F_member"] for chunk in member]) for member in chunks]
        ),
        "stress_prediction": torch.stack(
            [torch.cat([chunk["S_member"] for chunk in member]) for member in chunks]
        ),
        "energy_reference": torch.cat([chunk["E_ref"] for chunk in chunks[0]]),
        "forces_reference": torch.cat([chunk["F_ref"] for chunk in chunks[0]]),
        "stress_reference": torch.cat([chunk["S_ref"] for chunk in chunks[0]]),
        "n_atoms": torch.tensor(summary["n_atoms"], dtype=torch.int64),
        "structure_offsets": torch.tensor([0, 1, 3, 4], dtype=torch.int64),
        "member_ids": tuple(summary["member_ids"]),
        "structure_ids": tuple(summary["structure_ids"]),
        "atomic_numbers": torch.cat([chunk["atomic_numbers"] for chunk in chunks[0]]),
        "structure_mapping": torch.tensor([0, 1, 1, 2], dtype=torch.int64),
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
        "statistics": {"K": 2, "S": 3, "A": 4},
    }
    evaluated = evaluate_prediction(payload, (1.0, 0.5), 1e-12)
    torch.save(
        {
            "member_ids": list(payload["member_ids"]),
            "E_mean": evaluated.ensemble["energy"],
            "F_mean": evaluated.ensemble["forces"],
            "S_mean": evaluated.ensemble["stress"],
            "E_ref": payload["energy_reference"],
            "F_ref": payload["forces_reference"],
            "S_ref": payload["stress_reference"],
            "n_atoms": payload["n_atoms"],
            "atom_offsets": payload["structure_offsets"],
            "canonical_uncertainty": dict(evaluated.uncertainty),
        },
        root / "matpes_test_full/uncertainty/test/uncertainty.pt",
    )
    canonical_metrics = dict(evaluated.metrics)
    canonical_metrics["report_inputs"] = dict(evaluated.report_inputs)
    _write_json(
        root / "matpes_test_full/evaluation/test_final_metrics.json",
        canonical_metrics,
    )

    config_path = tmp_path / "config.yaml"
    config_payload = json.loads(json.dumps(config_payload))
    config_payload["paths"] = {
        "base_checkpoint": str(base),
        "train_data": str(tmp_path / "train.extxyz"),
        "val_data": str(tmp_path / "val.extxyz"),
        "test_data": str(tmp_path / "test.extxyz"),
        "output_root": str(tmp_path / "outputs"),
    }
    config_path.write_text(yaml.safe_dump(config_payload), encoding="utf-8")
    config = load_config(config_path)
    config = replace(
        config,
        training=replace(config.training, device="cpu"),
        identity=replace(
            config.identity,
            base_checkpoint_sha256=_sha(base),
            test_data_sha256="a" * 64,
        ),
        fge=replace(config.fge, member_count=2, cycles=2),
        evaluation=replace(config.evaluation, risk_coverages=(1.0, 0.5)),
    )
    expected = LegacyExpectations(
        member_count=2,
        chunk_count=3,
        member_sha256=member_hashes,
        test_data_sha256="a" * 64,
        manifest_path="matpes_test_full/fge_manifest.json",
        members_directory="successful/members",
        prediction_directory="matpes_test_full/predictions/test",
        uncertainty_path="matpes_test_full/uncertainty/test/uncertainty.pt",
        metrics_path="matpes_test_full/evaluation/test_final_metrics.json",
    )
    return root, base, config, expected
