from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

import pytest
import torch
import yaml
from torch.nn import functional as F

from Uncertainty_Quantification.ConfidenceHead.confidence_head.cache import (
    RawStructure,
    build_raw_cache,
)
from Uncertainty_Quantification.ConfidenceHead.confidence_head.config import (
    ConfidenceConfig,
)
from Uncertainty_Quantification.ConfidenceHead.confidence_head.metrics import (
    classification_metrics,
)


_WORKFLOWS = "Uncertainty_Quantification.ConfidenceHead.confidence_head.workflows"
verify_run = import_module(f"{_WORKFLOWS}.verify").verify_run
evaluate_run = import_module(f"{_WORKFLOWS}.evaluate").evaluate_run
train_run = import_module(f"{_WORKFLOWS}.train").train_run


def _structure(
    structure_id: int,
    num_atoms: int,
    *,
    force_overflow: bool = False,
    energy_overflow: bool = False,
) -> RawStructure:
    atom_index = torch.arange(num_atoms, dtype=torch.float32)
    force_prediction = torch.stack(
        (0.1 + atom_index, 0.2 + atom_index, 0.3 + atom_index),
        dim=1,
    )
    force_delta = torch.full((num_atoms, 3), 0.05)
    if force_overflow:
        force_delta[-1, -1] = 0.75
    energy_prediction = torch.tensor(float(structure_id) / 100.0)
    energy_delta = 0.8 if energy_overflow else 0.1 * num_atoms
    return RawStructure(
        structure_id=structure_id,
        atomic_numbers=torch.arange(1, num_atoms + 1, dtype=torch.int64),
        force_prediction=force_prediction,
        force_reference=force_prediction + force_delta,
        energy_prediction=energy_prediction,
        energy_reference=energy_prediction + energy_delta,
        force_features=torch.stack((atom_index, atom_index + 0.25), dim=1),
        energy_features=torch.stack(
            (atom_index + 10.0, atom_index + 20.0, atom_index + 30.0),
            dim=1,
        ),
    )


@pytest.fixture
def complete_cache(tmp_path: Path) -> Path:
    split_structures = {
        "train": [
            _structure(101, 2),
            _structure(102, 1, force_overflow=True, energy_overflow=True),
        ],
        "validation": [
            _structure(201, 2),
            _structure(202, 1, force_overflow=True, energy_overflow=True),
        ],
        "test": [
            _structure(301, 2),
            _structure(302, 1, force_overflow=True, energy_overflow=True),
        ],
    }
    split_identity = {
        name: {
            "sha256": f"{index:064x}",
            "structure_count": 2,
            "atom_count": 3,
            "force_component_count": 9,
        }
        for index, name in enumerate(split_structures, start=1)
    }
    identity_payload = {
        "checkpoint": {"sha256": "a" * 64},
        "outputs": {
            "force_prediction": "non_conservative_forces",
            "energy_prediction": "energy",
            "force_features": "mtt::aux::force_features",
            "energy_features": "mtt::aux::energy_features",
        },
        "features": {"force_dim": 2, "energy_dim": 3, "dtype": "float32"},
        "splits": split_identity,
    }
    return build_raw_cache(
        output_root=tmp_path / "caches",
        split_structures=split_structures,
        identity_payload=identity_payload,
        shard_max_atoms=2,
    )


@pytest.fixture
def config(tmp_path: Path) -> ConfidenceConfig:
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.write_bytes(b"synthetic checkpoint is identified but never loaded")
    data = {}
    for index, split in enumerate(("train", "validation", "test"), start=1):
        path = tmp_path / f"{split}.xyz"
        path.write_text(split, encoding="utf-8")
        data[split] = {"path": path, "expected_sha256": f"{index:064x}"}
    return ConfidenceConfig.model_validate(
        {
            "profile": "smoke",
            "checkpoint": {
                "path": checkpoint,
                "expected_sha256": "a" * 64,
            },
            "data": data,
            "binning": {
                "force_num_bins": 3,
                "force_max_error": 0.5,
                "energy_num_bins": 3,
                "energy_max_error": 0.3,
            },
            "model": {
                "hidden_dims": [4],
                "dropout": 0.0,
                "cumulant_order": 2,
                "signed_root": True,
            },
            "trainer": {
                "max_epochs": 2,
                "ema_beta": 0.95,
                "early_stopping_patience": 15,
                "min_delta": 0.0001,
                "min_epochs": 3,
            },
            "run": {
                "output_root": tmp_path / "outputs",
                "seed": 7,
                "device": "cpu",
                "amp": False,
            },
        }
    )


def _pearson(left: torch.Tensor, right: torch.Tensor) -> float:
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    return float(
        (left_centered * right_centered).sum()
        / torch.sqrt(left_centered.square().sum() * right_centered.square().sum())
    )


def test_classification_metrics_match_direct_hand_calculation() -> None:
    logits = torch.tensor(
        [
            [3.0, 1.0, -1.0],
            [-1.0, 3.0, 1.0],
            [-1.0, 1.0, 3.0],
        ]
    )
    labels = torch.tensor([0, 1, 2])
    observed = torch.tensor([0.1, 0.4, 3.5])
    representatives = torch.tensor([0.5, 1.5, 2.5])

    actual = classification_metrics(logits, labels, observed, representatives)

    probabilities = torch.softmax(logits, dim=-1)
    expected = probabilities @ representatives
    one_hot = F.one_hot(labels, num_classes=3).to(probabilities)
    expected_brier = ((probabilities - one_hot).square().sum(dim=-1)).mean()
    expected_nll = -torch.log(probabilities[torch.arange(len(labels)), labels]).mean()
    ranks = torch.arange(3, dtype=torch.float32)
    assert actual == pytest.approx(
        {
            "sample_count": 3,
            "accuracy": 1.0,
            "nll": float(expected_nll),
            "brier": float(expected_brier),
            "mean_observed_error": float(observed.mean()),
            "mean_expected_error": float(expected.mean()),
            "mae_expected_vs_observed": float(torch.abs(expected - observed).mean()),
            "pearson": _pearson(expected, observed),
            "spearman": _pearson(ranks, ranks),
            "overflow_count": 1,
            "overflow_fraction": 1 / 3,
        }
    )


def test_train_evaluate_and_verify_synthetic_complete_cache(
    tmp_path: Path,
    config: ConfidenceConfig,
    complete_cache: Path,
) -> None:
    run_dir = train_run(
        config,
        cache_manifest_path=complete_cache,
        run_name="synthetic",
    )

    assert run_dir == config.run.output_root / "runs" / "synthetic"
    expected_training_artifacts = {
        "resolved_config.yaml",
        "binning.json",
        "manifest.json",
        "checkpoints/best.pt",
        "checkpoints/last.pt",
        "logs/metrics.jsonl",
    }
    for relative in expected_training_artifacts:
        assert (run_dir / relative).is_file(), relative
    resolved = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())
    assert resolved["run"]["device"] == "cpu"
    metric_lines = (run_dir / "logs/metrics.jsonl").read_text().splitlines()
    assert len(metric_lines) == config.trainer.max_epochs
    assert all(json.loads(line)["val/total_loss_ema"] >= 0 for line in metric_lines)

    evaluation_dir = evaluate_run(
        run_dir,
        cache_manifest_path=complete_cache,
    )

    assert evaluation_dir == run_dir / "evaluation"
    evaluation_manifest = json.loads(
        (evaluation_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert evaluation_manifest["status"] == "complete"
    assert evaluation_manifest["checkpoint"]["path"] == "../checkpoints/best.pt"
    predictions = torch.load(
        evaluation_dir / "test_predictions.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert set(predictions) == {
        "force_logits",
        "force_labels",
        "force_observed_errors",
        "force_expected_errors",
        "energy_logits",
        "energy_labels",
        "energy_observed_errors",
        "energy_expected_errors",
        "structure_ids",
        "atom_offsets",
        "force_representatives",
        "energy_representatives",
    }
    assert predictions["force_logits"].shape == (3, 3, 3)
    assert predictions["force_labels"].shape == (3, 3)
    assert predictions["force_observed_errors"].shape == (3, 3)
    assert predictions["force_expected_errors"].shape == (3, 3)
    assert predictions["energy_logits"].shape == (2, 3)
    assert predictions["energy_labels"].shape == (2,)
    assert predictions["energy_observed_errors"].shape == (2,)
    assert predictions["energy_expected_errors"].shape == (2,)
    torch.testing.assert_close(
        predictions["structure_ids"],
        torch.tensor([301, 302], dtype=torch.int64),
    )
    torch.testing.assert_close(
        predictions["atom_offsets"],
        torch.tensor([0, 2, 3], dtype=torch.int64),
    )

    metrics = json.loads((evaluation_dir / "metrics.json").read_text())
    assert metrics["force"]["sample_count"] == 9
    assert metrics["energy"]["sample_count"] == 2
    assert metrics["force"]["overflow_count"] == 1
    assert metrics["force"]["overflow_fraction"] == pytest.approx(1 / 9)
    assert metrics["energy"]["overflow_count"] == 1
    assert metrics["energy"]["overflow_fraction"] == pytest.approx(1 / 2)
    assert (evaluation_dir / "force_bin_summary.csv").is_file()
    assert (evaluation_dir / "energy_bin_summary.csv").is_file()
    assert verify_run(run_dir, full=True)["status"] == "complete"

    image_suffixes = {".png", ".pdf", ".svg"}
    assert not [
        path for path in run_dir.rglob("*") if path.suffix.lower() in image_suffixes
    ]


def test_full_verify_rejects_a_hash_modified_declared_artifact(
    config: ConfidenceConfig,
    complete_cache: Path,
) -> None:
    run_dir = train_run(
        config,
        cache_manifest_path=complete_cache,
        run_name="tampered",
    )
    evaluation_dir = evaluate_run(
        run_dir,
        cache_manifest_path=complete_cache,
    )
    metrics_path = evaluation_dir / "metrics.json"
    metrics_path.write_text(
        metrics_path.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sha256|hash"):
        verify_run(run_dir, full=True)
