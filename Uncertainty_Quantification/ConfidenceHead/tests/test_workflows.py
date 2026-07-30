from __future__ import annotations

import copy
import hashlib
import json
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from pydantic import ValidationError
from torch.nn import functional as F

from Uncertainty_Quantification.ConfidenceHead.confidence_head.cache import (
    RawStructure,
    build_raw_cache,
)
from Uncertainty_Quantification.ConfidenceHead.confidence_head.config import (
    ConfidenceConfig,
    RunConfig,
    SchedulerConfig,
)
from Uncertainty_Quantification.ConfidenceHead.confidence_head.metrics import (
    classification_metrics,
)


_WORKFLOWS = "Uncertainty_Quantification.ConfidenceHead.confidence_head.workflows"
verify_run = import_module(f"{_WORKFLOWS}.verify").verify_run
evaluate_module = import_module(f"{_WORKFLOWS}.evaluate")
evaluate_run = evaluate_module.evaluate_run
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
            "force_features": "mtt::aux::non_conservative_forces_last_layer_features",
            "energy_features": "mtt::aux::energy_last_layer_features",
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


def _config_copy(
    config: ConfidenceConfig, **sections: dict[str, Any]
) -> ConfidenceConfig:
    raw = config.model_dump()
    for section, updates in sections.items():
        raw[section].update(updates)
    return ConfidenceConfig.model_validate(raw)


def _assert_nested_equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _publish_modified_predictions(
    run_dir: Path,
    predictions: dict[str, Any],
) -> None:
    evaluation_dir = run_dir / "evaluation"
    prediction_path = evaluation_dir / "test_predictions.pt"
    torch.save(predictions, prediction_path)
    evaluation_path = evaluation_dir / "manifest.json"
    evaluation_manifest = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation_manifest["artifacts"]["test_predictions.pt"]["sha256"] = _sha256(
        prediction_path
    )
    evaluation_path.write_text(
        json.dumps(evaluation_manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["evaluation/test_predictions.pt"]["sha256"] = _sha256(
        prediction_path
    )
    manifest["artifacts"]["evaluation/manifest.json"]["sha256"] = _sha256(
        evaluation_path
    )
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_stop_then_same_run_name_resumes_exactly_from_last(
    config: ConfidenceConfig,
    complete_cache: Path,
) -> None:
    config = _config_copy(config, trainer={"max_epochs": 3})
    continuous = train_run(
        config,
        cache_manifest_path=complete_cache,
        run_name="continuous",
    )
    resumed = train_run(
        config,
        cache_manifest_path=complete_cache,
        run_name="resumed",
        stop_after_epoch=0,
    )

    resumed = train_run(
        config,
        cache_manifest_path=complete_cache,
        run_name="resumed",
        resume_from=resumed / "checkpoints" / "last.pt",
    )

    continuous_snapshot = torch.load(
        continuous / "checkpoints" / "last.pt",
        map_location="cpu",
        weights_only=False,
    )
    resumed_snapshot = torch.load(
        resumed / "checkpoints" / "last.pt",
        map_location="cpu",
        weights_only=False,
    )
    for field in (
        "model",
        "optimizer",
        "scheduler",
        "epoch",
        "global_step",
        "learning_rate",
        "ema",
        "best",
        "best_epoch",
        "best_step",
        "bad_epochs",
        "sampler_rng_state",
    ):
        _assert_nested_equal(continuous_snapshot[field], resumed_snapshot[field])
    assert (continuous / "logs" / "metrics.jsonl").read_text() == (
        resumed / "logs" / "metrics.jsonl"
    ).read_text()
    assert json.loads((resumed / "manifest.json").read_text())["epochs_completed"] == 3


def test_evaluate_rejects_tampered_declared_checkpoint_before_torch_load(
    config: ConfidenceConfig,
    complete_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = train_run(
        config,
        cache_manifest_path=complete_cache,
        run_name="tampered-before-load",
    )
    checkpoint = run_dir / "checkpoints" / "best.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered")
    calls: list[Path] = []

    def forbidden_load(path: Path, **_: Any) -> Any:
        calls.append(Path(path))
        raise AssertionError("torch.load must not run before declared SHA verification")

    monkeypatch.setattr(evaluate_module.torch, "load", forbidden_load)
    with pytest.raises(ValueError, match="sha256|hash"):
        evaluate_run(run_dir, cache_manifest_path=complete_cache)
    assert calls == []


@pytest.mark.parametrize("location", ["undeclared", "outside"])
def test_evaluate_rejects_checkpoint_that_is_not_declared_and_confined(
    tmp_path: Path,
    config: ConfidenceConfig,
    complete_cache: Path,
    location: str,
) -> None:
    run_dir = train_run(
        config,
        cache_manifest_path=complete_cache,
        run_name=f"checkpoint-{location}",
    )
    source = run_dir / "checkpoints" / "best.pt"
    checkpoint = (
        run_dir / "checkpoints" / "undeclared.pt"
        if location == "undeclared"
        else tmp_path / "outside.pt"
    )
    checkpoint.write_bytes(source.read_bytes())

    with pytest.raises(ValueError, match="declared|run directory|checkpoint"):
        evaluate_run(
            run_dir,
            cache_manifest_path=complete_cache,
            checkpoint_path=checkpoint,
        )


@pytest.mark.parametrize(
    ("section", "updates", "message"),
    [
        ("checkpoint", {"expected_sha256": "f" * 64}, "checkpoint"),
        (
            "data",
            {
                "test": {
                    "path": Path("test.xyz"),
                    "expected_sha256": "e" * 64,
                }
            },
            "test|split|data",
        ),
        (
            "readouts",
            {"force_features": "mtt::aux::different_force_features"},
            "readout|force_features|output",
        ),
    ],
)
def test_train_rejects_config_cache_identity_mismatch(
    config: ConfidenceConfig,
    complete_cache: Path,
    section: str,
    updates: dict[str, Any],
    message: str,
) -> None:
    mismatched = _config_copy(config, **{section: updates})

    with pytest.raises(ValueError, match=message):
        train_run(
            mismatched,
            cache_manifest_path=complete_cache,
            run_name=f"mismatch-{section}",
        )


def test_metadata_verify_checks_predictions_and_requires_complete_cache(
    config: ConfidenceConfig,
    complete_cache: Path,
) -> None:
    run_dir = train_run(
        config,
        cache_manifest_path=complete_cache,
        run_name="metadata-verification",
    )
    evaluation_dir = evaluate_run(run_dir, cache_manifest_path=complete_cache)
    predictions = torch.load(
        evaluation_dir / "test_predictions.pt",
        map_location="cpu",
        weights_only=True,
    )
    predictions["force_labels"] = predictions["force_labels"][:-1]
    _publish_modified_predictions(run_dir, predictions)

    with pytest.raises(ValueError, match="shape|count"):
        verify_run(run_dir, full=False)

    evaluate_run(run_dir, cache_manifest_path=complete_cache)
    cache_manifest = json.loads(complete_cache.read_text(encoding="utf-8"))
    cache_manifest["status"] = "incomplete"
    complete_cache.write_text(
        json.dumps(cache_manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cache.*complete|complete.*cache"):
        verify_run(run_dir, full=False)


def test_distinct_force_and_energy_bin_counts_train_and_evaluate(
    config: ConfidenceConfig,
    complete_cache: Path,
) -> None:
    config = _config_copy(
        config,
        trainer={"max_epochs": 1},
        binning={"force_num_bins": 3, "energy_num_bins": 5},
    )
    run_dir = train_run(
        config,
        cache_manifest_path=complete_cache,
        run_name="independent-bins",
    )
    evaluation_dir = evaluate_run(run_dir, cache_manifest_path=complete_cache)
    predictions = torch.load(
        evaluation_dir / "test_predictions.pt",
        map_location="cpu",
        weights_only=True,
    )

    assert predictions["force_logits"].shape == (3, 3, 3)
    assert predictions["energy_logits"].shape == (2, 5)
    assert predictions["force_representatives"].shape == (3,)
    assert predictions["energy_representatives"].shape == (5,)


def test_amp_is_explicitly_unsupported_on_every_device() -> None:
    for device in ("cpu", "cuda"):
        with pytest.raises(ValidationError, match="amp"):
            RunConfig(device=device, amp=True)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("monitor", "other"),
        ("factor", 0.25),
        ("patience", 4),
        ("threshold", 0.0),
        ("threshold_mode", "rel"),
        ("cooldown", 1),
        ("min_lr", 0.0),
    ],
)
def test_scheduler_configuration_is_fixed_to_approved_constants(
    field: str,
    value: Any,
) -> None:
    assert SchedulerConfig().model_dump() == {
        "monitor": "val/total_loss_ema",
        "factor": 0.5,
        "patience": 5,
        "threshold": 0.0001,
        "threshold_mode": "abs",
        "cooldown": 0,
        "min_lr": 0.000001,
    }
    with pytest.raises(ValidationError, match=field):
        SchedulerConfig.model_validate({field: value})


def test_max_epochs_stop_reason_is_recorded_in_manifest_and_last_checkpoint(
    config: ConfidenceConfig,
    complete_cache: Path,
) -> None:
    config = _config_copy(config, trainer={"max_epochs": 1})
    run_dir = train_run(
        config,
        cache_manifest_path=complete_cache,
        run_name="max-epochs-reason",
    )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    snapshot = torch.load(
        run_dir / "checkpoints" / "last.pt",
        map_location="cpu",
        weights_only=False,
    )

    assert manifest["stop_reason"] == "max_epochs"
    assert snapshot["stop_reason"] == "max_epochs"


def test_verify_rejects_invalid_prediction_semantics(
    config: ConfidenceConfig,
    complete_cache: Path,
) -> None:
    run_dir = train_run(
        config,
        cache_manifest_path=complete_cache,
        run_name="prediction-semantics",
    )
    evaluation_dir = evaluate_run(run_dir, cache_manifest_path=complete_cache)
    prediction_path = evaluation_dir / "test_predictions.pt"
    original = torch.load(
        prediction_path,
        map_location="cpu",
        weights_only=True,
    )
    mutations = {
        "dtype": lambda value: value.update(
            force_logits=value["force_logits"].to(torch.int64)
        ),
        "label range": lambda value: value["energy_labels"].fill_(99),
        "unique structure IDs": lambda value: value["structure_ids"].fill_(301),
        "offset": lambda value: value["atom_offsets"].__setitem__(1, 0),
        "representatives": lambda value: value["force_representatives"].flip(0),
        "expected error": lambda value: value["energy_expected_errors"].add_(1.0),
    }
    for message, mutate in mutations.items():
        candidate = copy.deepcopy(original)
        mutate(candidate)
        _publish_modified_predictions(run_dir, candidate)
        with pytest.raises(ValueError, match=message):
            verify_run(run_dir, full=False)
