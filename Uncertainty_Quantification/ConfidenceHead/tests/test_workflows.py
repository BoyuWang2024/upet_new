from __future__ import annotations

import copy
import hashlib
import io
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
train_module = import_module(f"{_WORKFLOWS}.train")
train_run = train_module.train_run


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
        "unique.*structure IDs|structure IDs.*unique": lambda value: value[
            "structure_ids"
        ].fill_(301),
        "offset": lambda value: value["atom_offsets"].__setitem__(1, 0),
        "representatives": lambda value: value.update(
            force_representatives=value["force_representatives"].flip(0)
        ),
        "expected error": lambda value: value["energy_expected_errors"].add_(1.0),
    }
    for message, mutate in mutations.items():
        candidate = copy.deepcopy(original)
        mutate(candidate)
        _publish_modified_predictions(run_dir, candidate)
        with pytest.raises(ValueError, match=message):
            verify_run(run_dir, full=False)


def _resume_candidate(
    config: ConfidenceConfig,
    complete_cache: Path,
    run_name: str,
) -> tuple[ConfidenceConfig, Path]:
    resumable = _config_copy(config, trainer={"max_epochs": 3})
    run_dir = train_run(
        resumable,
        cache_manifest_path=complete_cache,
        run_name=run_name,
        stop_after_epoch=0,
    )
    return resumable, run_dir


def _update_declared_hash(run_dir: Path, artifact: str) -> None:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][artifact]["sha256"] = _sha256(run_dir / artifact)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_checkpoint_consumers_load_verified_stable_bytes_not_paths(
    config: ConfidenceConfig,
    complete_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resumable, run_dir = _resume_candidate(
        config,
        complete_cache,
        "stable-checkpoint-bytes",
    )
    original_load = torch.load
    unsafe_sources: list[Any] = []

    def record_load(source: Any, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("weights_only") is False:
            unsafe_sources.append(source)
        return original_load(source, *args, **kwargs)

    monkeypatch.setattr(torch, "load", record_load)
    train_run(
        resumable,
        cache_manifest_path=complete_cache,
        run_name="stable-checkpoint-bytes",
        resume_from=run_dir / "checkpoints" / "last.pt",
    )
    evaluate_run(run_dir, cache_manifest_path=complete_cache)
    verify_run(run_dir, full=True)

    assert len(unsafe_sources) >= 4
    assert all(isinstance(source, io.BytesIO) for source in unsafe_sources)


def test_resume_rejects_traversal_run_name_before_following_resume_path(
    config: ConfidenceConfig,
    complete_cache: Path,
) -> None:
    resumable, run_dir = _resume_candidate(
        config,
        complete_cache,
        "safe-resume-name",
    )

    with pytest.raises(ValueError, match="unsafe run name"):
        train_run(
            resumable,
            cache_manifest_path=complete_cache,
            run_name="../escaped",
            resume_from=run_dir / "checkpoints" / "last.pt",
        )


@pytest.mark.parametrize(
    "failure",
    ["checkpoint_hash", "checkpoint_payload", "metrics_hash"],
)
def test_failed_resume_leaves_previous_manifest_bytes_unchanged(
    config: ConfidenceConfig,
    complete_cache: Path,
    failure: str,
) -> None:
    resumable, run_dir = _resume_candidate(
        config,
        complete_cache,
        f"immutable-manifest-{failure}",
    )
    checkpoint = run_dir / "checkpoints" / "last.pt"
    metrics_path = run_dir / "logs" / "metrics.jsonl"
    if failure == "checkpoint_hash":
        checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered")
    elif failure == "checkpoint_payload":
        torch.save({"schema_version": "invalid"}, checkpoint)
        _update_declared_hash(run_dir, "checkpoints/last.pt")
    else:
        metrics_path.write_text(
            metrics_path.read_text(encoding="utf-8") + " ",
            encoding="utf-8",
        )
    manifest_path = run_dir / "manifest.json"
    before = manifest_path.read_bytes()

    with pytest.raises((ValueError, RuntimeError), match="checkpoint|metrics|sha256"):
        train_run(
            resumable,
            cache_manifest_path=complete_cache,
            run_name=f"immutable-manifest-{failure}",
            resume_from=checkpoint,
        )
    assert manifest_path.read_bytes() == before


@pytest.mark.parametrize(
    "failure",
    ["declared_sha", "epoch_gap", "snapshot_mismatch"],
)
def test_resume_validates_metrics_integrity_and_snapshot_boundary(
    config: ConfidenceConfig,
    complete_cache: Path,
    failure: str,
) -> None:
    resumable, run_dir = _resume_candidate(
        config,
        complete_cache,
        f"invalid-resume-metrics-{failure}",
    )
    metrics_path = run_dir / "logs" / "metrics.jsonl"
    records = [
        json.loads(line) for line in metrics_path.read_text().splitlines() if line
    ]
    if failure == "declared_sha":
        records[0]["train/total_loss"] += 1.0
    elif failure == "epoch_gap":
        records[0]["epoch"] = 2
    else:
        records[-1]["global_step"] += 1
        records[-1]["learning_rate"] *= 0.5
        records[-1]["val/total_loss_ema"] += 1.0
    metrics_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    if failure != "declared_sha":
        _update_declared_hash(run_dir, "logs/metrics.jsonl")

    with pytest.raises(
        ValueError, match="metrics.*sha256|epoch|global_step|learning_rate|EMA"
    ):
        train_run(
            resumable,
            cache_manifest_path=complete_cache,
            run_name=f"invalid-resume-metrics-{failure}",
            resume_from=run_dir / "checkpoints" / "last.pt",
        )


def test_stop_after_epoch_on_final_epoch_remains_external_stop_in_last(
    config: ConfidenceConfig,
    complete_cache: Path,
) -> None:
    one_epoch = _config_copy(config, trainer={"max_epochs": 1})
    run_dir = train_run(
        one_epoch,
        cache_manifest_path=complete_cache,
        run_name="external-stop-on-final",
        stop_after_epoch=0,
    )
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    last = torch.load(
        run_dir / "checkpoints" / "last.pt",
        map_location="cpu",
        weights_only=False,
    )

    assert manifest["stop_reason"] == "external_stop_after_epoch"
    assert last["stop_reason"] == "external_stop_after_epoch"


def test_verify_recomputes_cache_identity_from_schema_and_payload(
    config: ConfidenceConfig,
    complete_cache: Path,
) -> None:
    run_dir = train_run(
        config,
        cache_manifest_path=complete_cache,
        run_name="recomputed-cache-identity",
    )
    evaluate_run(run_dir, cache_manifest_path=complete_cache)
    cache = json.loads(complete_cache.read_text(encoding="utf-8"))
    cache["identity_payload"]["checkpoint"]["sha256"] = "f" * 64
    complete_cache.write_text(
        json.dumps(cache, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cache.*identity|identity.*cache"):
        verify_run(run_dir, full=False)


def test_verify_rejects_incomplete_evaluation_declaration(
    config: ConfidenceConfig,
    complete_cache: Path,
) -> None:
    run_dir = train_run(
        config,
        cache_manifest_path=complete_cache,
        run_name="incomplete-evaluation",
    )
    evaluate_run(run_dir, cache_manifest_path=complete_cache)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["evaluation"]["status"] = "incomplete"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="evaluation.*complete|complete.*evaluation"):
        verify_run(run_dir, full=False)


def test_verify_binds_representatives_exactly_to_binning_artifact(
    config: ConfidenceConfig,
    complete_cache: Path,
) -> None:
    run_dir = train_run(
        config,
        cache_manifest_path=complete_cache,
        run_name="representatives-bound-to-binning",
    )
    evaluation_dir = evaluate_run(run_dir, cache_manifest_path=complete_cache)
    predictions = torch.load(
        evaluation_dir / "test_predictions.pt",
        map_location="cpu",
        weights_only=True,
    )
    predictions["force_representatives"] = predictions["force_representatives"] + 0.01
    predictions["force_expected_errors"] = (
        torch.softmax(predictions["force_logits"], dim=-1)
        @ predictions["force_representatives"]
    )
    _publish_modified_predictions(run_dir, predictions)

    with pytest.raises(
        ValueError, match="representatives.*binning|binning.*representatives"
    ):
        verify_run(run_dir, full=False)


@pytest.mark.parametrize(
    "identity",
    ["config_id", "binning_id", "model_loss_id", "run_id"],
)
def test_evaluate_rejects_coordinated_artifact_identity_tampering_before_load(
    config: ConfidenceConfig,
    complete_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
    identity: str,
) -> None:
    run_dir = train_run(
        config,
        cache_manifest_path=complete_cache,
        run_name=f"evaluate-rederive-{identity}",
    )
    if identity in {"config_id", "model_loss_id"}:
        config_path = run_dir / "resolved_config.yaml"
        resolved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if identity == "config_id":
            resolved["run"]["seed"] += 1
        else:
            resolved["loss"]["force_coefficient"] += 0.25
        config_path.write_text(
            yaml.safe_dump(resolved, sort_keys=True),
            encoding="utf-8",
        )
        _update_declared_hash(run_dir, "resolved_config.yaml")
    elif identity == "binning_id":
        binning_path = run_dir / "binning.json"
        binning = json.loads(binning_path.read_text(encoding="utf-8"))
        binning["force"]["representatives"][0] += 0.01
        binning_path.write_text(
            json.dumps(binning, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _update_declared_hash(run_dir, "binning.json")
    else:
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["run_id"] = "run-" + "f" * 16
        manifest["identity"] = manifest["run_id"]
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    calls: list[Any] = []

    def forbidden_load(source: Any, **_: Any) -> Any:
        calls.append(source)
        raise AssertionError("identity must be rejected before checkpoint load")

    monkeypatch.setattr(evaluate_module.torch, "load", forbidden_load)
    with pytest.raises(
        ValueError, match="identity|config_id|binning_id|model_loss_id|run"
    ):
        evaluate_run(run_dir, cache_manifest_path=complete_cache)
    assert calls == []


def test_run_and_evaluation_provenance_cover_dependencies_times_and_resume_history(
    config: ConfidenceConfig,
    complete_cache: Path,
) -> None:
    resumable, run_dir = _resume_candidate(
        config,
        complete_cache,
        "provenance-history",
    )
    initial_manifest = json.loads(
        (run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    initial_provenance = initial_manifest["provenance"]
    assert {"upet", "metatomic"} <= set(initial_provenance["dependencies"])
    initial_started_at = initial_provenance["started_at"]
    assert initial_started_at
    assert initial_provenance["completed_at"]

    train_run(
        resumable,
        cache_manifest_path=complete_cache,
        run_name="provenance-history",
        resume_from=run_dir / "checkpoints" / "last.pt",
    )
    resumed_provenance = json.loads(
        (run_dir / "manifest.json").read_text(encoding="utf-8")
    )["provenance"]
    assert resumed_provenance["started_at"] == initial_started_at
    assert isinstance(resumed_provenance["history"], list)
    assert resumed_provenance["history"]
    assert all(
        entry["started_at"] and entry["completed_at"]
        for entry in resumed_provenance["history"]
    )

    evaluation_dir = evaluate_run(run_dir, cache_manifest_path=complete_cache)
    evaluation_manifest = json.loads(
        (evaluation_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert evaluation_manifest["started_at"]
    assert evaluation_manifest["completed_at"]


def _resume_transaction_bytes(run_dir: Path) -> dict[str, bytes | None]:
    return {
        relative: path.read_bytes() if path.is_file() else None
        for relative in (
            "manifest.json",
            "resolved_config.yaml",
            "binning.json",
            "checkpoints/best.pt",
            "checkpoints/last.pt",
            "logs/metrics.jsonl",
        )
        for path in (run_dir / relative,)
    }


@pytest.mark.parametrize(
    "failure",
    [
        "manifest_status",
        "resolved_config",
        "binning",
        "best_missing",
        "best_hash",
    ],
)
def test_resume_prevalidates_all_previous_artifacts_before_any_write(
    config: ConfidenceConfig,
    complete_cache: Path,
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resumable, run_dir = _resume_candidate(
        config,
        complete_cache,
        f"resume-preflight-{failure}",
    )
    manifest_path = run_dir / "manifest.json"
    if failure == "manifest_status":
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["status"] = "incomplete"
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    elif failure == "resolved_config":
        path = run_dir / "resolved_config.yaml"
        resolved = yaml.safe_load(path.read_text(encoding="utf-8"))
        resolved["run"]["seed"] += 1
        path.write_text(yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8")
        _update_declared_hash(run_dir, "resolved_config.yaml")
    elif failure == "binning":
        path = run_dir / "binning.json"
        bins = json.loads(path.read_text(encoding="utf-8"))
        bins["force"]["representatives"][0] += 0.01
        path.write_text(
            json.dumps(bins, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _update_declared_hash(run_dir, "binning.json")
    elif failure == "best_missing":
        (run_dir / "checkpoints" / "best.pt").unlink()
    else:
        best = run_dir / "checkpoints" / "best.pt"
        best.write_bytes(best.read_bytes() + b"tampered")
    writes: list[Path] = []
    original_atomic_write_bytes = train_module._atomic_write_bytes

    def record_atomic_write(path: Path, data: bytes) -> None:
        writes.append(Path(path))
        original_atomic_write_bytes(path, data)

    monkeypatch.setattr(train_module, "_atomic_write_bytes", record_atomic_write)
    before = _resume_transaction_bytes(run_dir)

    with pytest.raises(
        (OSError, RuntimeError, ValueError), match="resume|artifact|run"
    ):
        train_run(
            resumable,
            cache_manifest_path=complete_cache,
            run_name=f"resume-preflight-{failure}",
            resume_from=run_dir / "checkpoints" / "last.pt",
        )

    assert _resume_transaction_bytes(run_dir) == before
    assert writes == []


def test_resume_prevalidates_declared_evaluation_artifacts_before_any_write(
    config: ConfidenceConfig,
    complete_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resumable, run_dir = _resume_candidate(
        config,
        complete_cache,
        "resume-evaluation-preflight",
    )
    evaluation_dir = evaluate_run(run_dir, cache_manifest_path=complete_cache)
    metrics = evaluation_dir / "metrics.json"
    metrics.write_bytes(metrics.read_bytes() + b"tampered")
    writes: list[Path] = []
    original_atomic_write_bytes = train_module._atomic_write_bytes

    def record_atomic_write(path: Path, data: bytes) -> None:
        writes.append(Path(path))
        original_atomic_write_bytes(path, data)

    monkeypatch.setattr(train_module, "_atomic_write_bytes", record_atomic_write)

    with pytest.raises(ValueError, match="evaluation/metrics.json|sha256"):
        train_run(
            resumable,
            cache_manifest_path=complete_cache,
            run_name="resume-evaluation-preflight",
            resume_from=run_dir / "checkpoints" / "last.pt",
        )

    assert writes == []


def test_successful_resume_invalidates_declared_evaluation_artifacts(
    config: ConfidenceConfig,
    complete_cache: Path,
) -> None:
    resumable, run_dir = _resume_candidate(
        config,
        complete_cache,
        "resume-invalidates-evaluation",
    )
    evaluate_run(run_dir, cache_manifest_path=complete_cache)
    before = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    evaluation_relatives = {
        relative
        for relative in before["artifacts"]
        if relative.startswith("evaluation/")
    }
    assert evaluation_relatives

    train_run(
        resumable,
        cache_manifest_path=complete_cache,
        run_name="resume-invalidates-evaluation",
        resume_from=run_dir / "checkpoints" / "last.pt",
    )

    after = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "evaluation" not in after
    assert not {
        relative
        for relative in after["artifacts"]
        if relative.startswith("evaluation/")
    }
    assert all(not (run_dir / relative).exists() for relative in evaluation_relatives)


def test_resume_parses_the_same_metrics_bytes_whose_digest_was_verified(
    config: ConfidenceConfig,
    complete_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resumable, run_dir = _resume_candidate(
        config,
        complete_cache,
        "resume-stable-metrics-bytes",
    )
    metrics_path = (run_dir / "logs" / "metrics.jsonl").resolve()
    original_sha256_file = train_module.sha256_file
    original_read_bytes = Path.read_bytes
    swapped = False
    metrics_reads = 0

    def replace_after_path_hash(path: Path) -> str:
        nonlocal swapped
        digest = original_sha256_file(path)
        if Path(path).resolve() == metrics_path and not swapped:
            swapped = True
            metrics_path.write_text("{not valid json\n", encoding="utf-8")
        return digest

    def replace_after_bytes_read(path: Path) -> bytes:
        nonlocal metrics_reads, swapped
        data = original_read_bytes(path)
        if path.resolve() == metrics_path:
            metrics_reads += 1
        # Rollback snapshots start only after preflight; race the verified read.
        if path.resolve() == metrics_path and metrics_reads == 1 and not swapped:
            swapped = True
            metrics_path.write_text("{not valid json\n", encoding="utf-8")
        return data

    monkeypatch.setattr(train_module, "sha256_file", replace_after_path_hash)
    monkeypatch.setattr(Path, "read_bytes", replace_after_bytes_read)
    train_run(
        resumable,
        cache_manifest_path=complete_cache,
        run_name="resume-stable-metrics-bytes",
        resume_from=run_dir / "checkpoints" / "last.pt",
    )
    assert swapped


@pytest.mark.parametrize("mode", ["runs_root", "resume_run"])
def test_run_directory_symlinks_cannot_escape_output_root(
    config: ConfidenceConfig,
    complete_cache: Path,
    tmp_path: Path,
    mode: str,
) -> None:
    outside = tmp_path / f"outside-{mode}"
    outside.mkdir()
    run_name = f"symlink-escape-{mode}"
    if mode == "runs_root":
        runs_root = config.run.output_root / "runs"
        runs_root.parent.mkdir(parents=True, exist_ok=True)
        runs_root.symlink_to(outside, target_is_directory=True)
        resume_from = None
        training_config = config
    else:
        training_config, run_dir = _resume_candidate(
            config,
            complete_cache,
            run_name,
        )
        moved = outside / run_name
        run_dir.rename(moved)
        run_dir.symlink_to(moved, target_is_directory=True)
        resume_from = run_dir / "checkpoints" / "last.pt"

    with pytest.raises(ValueError, match="symlink|escape|unsafe"):
        train_run(
            training_config,
            cache_manifest_path=complete_cache,
            run_name=run_name,
            resume_from=resume_from,
        )


@pytest.mark.parametrize("failure", ["non_mapping", "identity"])
def test_verify_rejects_malformed_or_mismatched_evaluation_identity(
    config: ConfidenceConfig,
    complete_cache: Path,
    failure: str,
) -> None:
    run_dir = train_run(
        config,
        cache_manifest_path=complete_cache,
        run_name=f"evaluation-declaration-{failure}",
    )
    evaluation_dir = evaluate_run(run_dir, cache_manifest_path=complete_cache)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if failure == "non_mapping":
        manifest["evaluation"] = "complete"
    else:
        evaluation_path = evaluation_dir / "manifest.json"
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        evaluation["identity"] = "run-" + "f" * 16
        evaluation_path.write_text(
            json.dumps(evaluation, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest["artifacts"]["evaluation/manifest.json"]["sha256"] = _sha256(
            evaluation_path
        )
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="evaluation.*identity|evaluation.*mapping"):
        verify_run(run_dir, full=False)


def test_verify_binds_evaluation_checkpoint_to_declared_run_checkpoint(
    config: ConfidenceConfig,
    complete_cache: Path,
) -> None:
    run_dir = train_run(
        config,
        cache_manifest_path=complete_cache,
        run_name="evaluation-checkpoint-binding",
    )
    evaluation_dir = evaluate_run(run_dir, cache_manifest_path=complete_cache)
    evaluation_path = evaluation_dir / "manifest.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    replacement = run_dir / "resolved_config.yaml"
    evaluation["checkpoint"] = {
        "path": "../resolved_config.yaml",
        "sha256": _sha256(replacement),
    }
    evaluation_path.write_text(
        json.dumps(evaluation, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["evaluation/manifest.json"]["sha256"] = _sha256(
        evaluation_path
    )
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="evaluation checkpoint.*declared"):
        verify_run(run_dir, full=False)


def test_evaluation_provenance_uses_the_checkpoint_digest_that_was_loaded(
    config: ConfidenceConfig,
    complete_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = train_run(
        config,
        cache_manifest_path=complete_cache,
        run_name="evaluation-loaded-checkpoint-digest",
    )
    checkpoint = run_dir / "checkpoints" / "best.pt"
    loaded_digest = _sha256(checkpoint)
    original_loader = evaluate_module.load_verified_torch

    def replace_after_verified_load(*args: Any, **kwargs: Any) -> Any:
        snapshot = original_loader(*args, **kwargs)
        checkpoint.write_bytes(b"replacement after verified load")
        return snapshot

    monkeypatch.setattr(
        evaluate_module,
        "load_verified_torch",
        replace_after_verified_load,
    )
    evaluation_dir = evaluate_run(run_dir, cache_manifest_path=complete_cache)
    evaluation = json.loads(
        (evaluation_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert evaluation["checkpoint"]["sha256"] == loaded_digest


def test_run_lock_rejects_a_second_writer_before_artifact_changes(
    config: ConfidenceConfig,
    complete_cache: Path,
) -> None:
    resumable, run_dir = _resume_candidate(
        config,
        complete_cache,
        "exclusive-run-writer",
    )
    before = _resume_transaction_bytes(run_dir)

    with train_module._exclusive_run_lock(
        resumable.run.output_root,
        "exclusive-run-writer",
    ):
        with pytest.raises(ValueError, match="lock|writer|active"):
            train_run(
                resumable,
                cache_manifest_path=complete_cache,
                run_name="exclusive-run-writer",
                resume_from=run_dir / "checkpoints" / "last.pt",
            )
        assert _resume_transaction_bytes(run_dir) == before

    train_run(
        resumable,
        cache_manifest_path=complete_cache,
        run_name="exclusive-run-writer",
        resume_from=run_dir / "checkpoints" / "last.pt",
    )


@pytest.mark.parametrize("artifact", ["resolved_config.yaml", "binning.json"])
def test_resume_rolls_back_if_static_artifact_changes_after_preflight(
    config: ConfidenceConfig,
    complete_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
) -> None:
    run_name = f"resume-post-preflight-{artifact.split('.')[0]}"
    resumable, run_dir = _resume_candidate(config, complete_cache, run_name)
    before = _resume_transaction_bytes(run_dir)
    original_epoch = train_module._epoch
    mutated = False

    def mutate_after_first_train_epoch(**kwargs: Any) -> Any:
        nonlocal mutated
        result = original_epoch(**kwargs)
        if kwargs["optimizer"] is not None and not mutated:
            mutated = True
            path = run_dir / artifact
            if artifact.endswith(".yaml"):
                payload = yaml.safe_load(path.read_text(encoding="utf-8"))
                payload["run"]["seed"] += 1
                path.write_text(
                    yaml.safe_dump(payload, sort_keys=True),
                    encoding="utf-8",
                )
            else:
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["energy"]["representatives"][0] += 0.01
                path.write_text(
                    json.dumps(payload, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
        return result

    monkeypatch.setattr(train_module, "_epoch", mutate_after_first_train_epoch)
    with pytest.raises(ValueError, match="changed|artifact|publish"):
        train_run(
            resumable,
            cache_manifest_path=complete_cache,
            run_name=run_name,
            resume_from=run_dir / "checkpoints" / "last.pt",
        )
    assert mutated
    assert _resume_transaction_bytes(run_dir) == before


def test_resume_rolls_back_if_best_changes_after_preflight(
    config: ConfidenceConfig,
    complete_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_name = "resume-post-preflight-best"
    resumable, run_dir = _resume_candidate(config, complete_cache, run_name)
    before = _resume_transaction_bytes(run_dir)
    original_commit = train_module.commit_epoch_checkpoints
    mutated = False

    def mutate_best_after_checkpoint_commit(**kwargs: Any) -> bool:
        nonlocal mutated
        external_stop = original_commit(**kwargs)
        mutated = True
        (run_dir / "checkpoints" / "best.pt").write_bytes(b"replaced best")
        return external_stop

    monkeypatch.setattr(
        train_module,
        "commit_epoch_checkpoints",
        mutate_best_after_checkpoint_commit,
    )
    with pytest.raises((RuntimeError, ValueError), match="best|checkpoint|artifact"):
        train_run(
            resumable,
            cache_manifest_path=complete_cache,
            run_name=run_name,
            resume_from=run_dir / "checkpoints" / "last.pt",
        )
    assert mutated
    assert _resume_transaction_bytes(run_dir) == before
