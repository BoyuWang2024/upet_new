from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml
from confidence_head.binning import fixed_linear_binning
from confidence_head.external_prediction import (
    _publish_external_result,
    discover_external_runs,
    verify_external_prediction,
)
from confidence_head.single_target_prediction import SingleTargetResult


def _write_run(root: Path, name: str, *, target: str, order: int = 3) -> Path:
    run = root / name
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(
        json.dumps({"status": "complete", "run_id": f"run-{name}"}),
        encoding="utf-8",
    )
    force_weight = 1.0 if target == "force" else 0.0
    energy_weight = 1.0 if target == "energy" else 0.0
    (run / "resolved_config.yaml").write_text(
        yaml.safe_dump(
            {
                "loss": {
                    "force_coefficient": force_weight,
                    "energy_coefficient": energy_weight,
                },
                "model": {
                    "force": {"enabled": True, "target_mode": "atom_mean"},
                    "energy": {"enabled": True, "cumulant_order": order},
                },
            }
        ),
        encoding="utf-8",
    )
    return run


def _nine_runs(tmp_path: Path) -> Path:
    root = tmp_path / "runs"
    for order in range(1, 9):
        _write_run(root, f"energy-{order}", target="energy", order=order)
    _write_run(root, "force", target="force")
    return root


def test_discovery_requires_exact_energy_orders_and_one_atom_mean_force(
    tmp_path: Path,
) -> None:
    runs = discover_external_runs(_nine_runs(tmp_path))

    assert set(runs.energy_by_order) == set(range(1, 9))
    assert runs.force.name == "force"


def test_discovery_rejects_duplicate_energy_order(tmp_path: Path) -> None:
    root = _nine_runs(tmp_path)
    _write_run(root, "energy-1-copy", target="energy", order=1)

    with pytest.raises(ValueError, match="duplicate.*order 1"):
        discover_external_runs(root)


def _energy_result() -> SingleTargetResult:
    return SingleTargetResult(
        target="energy",
        predictions={
            "structure_ids": torch.tensor([4, 5], dtype=torch.int64),
            "atom_offsets": torch.tensor([0, 2, 3], dtype=torch.int64),
            "energy_prediction": torch.tensor([1.0, 2.0]),
            "energy_reference": torch.tensor([1.2, 2.1]),
            "energy_logits": torch.tensor(
                [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0]], dtype=torch.float32
            ),
            "energy_labels": torch.tensor([0, 1], dtype=torch.int64),
            "energy_observed_errors": torch.tensor([0.1, 0.1]),
            "energy_expected_errors": torch.tensor([0.2, 1.0]),
            "energy_representatives": torch.tensor([0.5, 1.5, 2.5]),
        },
        metrics={"energy": {"mae_expected_error": 0.5}},
    )


def test_publication_is_verified_and_does_not_mutate_run_manifest(
    tmp_path: Path,
) -> None:
    run = tmp_path / "outputs" / "runs" / "energy-1"
    run.mkdir(parents=True)
    run_manifest = run / "manifest.json"
    run_manifest.write_text('{"run_id":"run-energy-1","status":"complete"}\n')
    before = run_manifest.read_bytes()
    spec = fixed_linear_binning(3, 3.0)

    result = _publish_external_result(
        run_dir=run,
        dataset_name="mad_test",
        result=_energy_result(),
        spec=spec,
        identity_payload={
            "dataset": {"name": "mad_test", "sha256": "1" * 64},
            "source_run_id": "run-energy-1",
            "head_checkpoint_sha256": "2" * 64,
            "input_cache_id": "cache-external",
            "training_cache_id": "cache-training",
            "binning_id": "binning-1",
            "target": "energy",
            "order": 1,
        },
    )

    manifest = verify_external_prediction(result / "manifest.json", full=True)
    assert manifest["status"] == "complete"
    assert manifest["dataset"]["name"] == "mad_test"
    assert run_manifest.read_bytes() == before
    assert (result / "energy_bin_summary.csv").is_file()


def test_publication_reuses_identical_and_rejects_conflicting_identity(
    tmp_path: Path,
) -> None:
    run = tmp_path / "outputs" / "runs" / "energy-1"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(
        '{"run_id":"run-energy-1","status":"complete"}\n', encoding="utf-8"
    )
    spec = fixed_linear_binning(3, 3.0)
    payload = {
        "dataset": {"name": "mad_test", "sha256": "1" * 64},
        "source_run_id": "run-energy-1",
        "head_checkpoint_sha256": "2" * 64,
        "input_cache_id": "cache-external",
        "training_cache_id": "cache-training",
        "binning_id": "binning-1",
        "target": "energy",
        "order": 1,
    }
    first = _publish_external_result(
        run_dir=run,
        dataset_name="mad_test",
        result=_energy_result(),
        spec=spec,
        identity_payload=payload,
    )
    second = _publish_external_result(
        run_dir=run,
        dataset_name="mad_test",
        result=_energy_result(),
        spec=spec,
        identity_payload=payload,
    )
    assert second == first

    with pytest.raises(ValueError, match="identity"):
        _publish_external_result(
            run_dir=run,
            dataset_name="mad_test",
            result=_energy_result(),
            spec=spec,
            identity_payload={**payload, "head_checkpoint_sha256": "3" * 64},
        )
