from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from Uncertainty_Quantification.ConfidenceHead.confidence_head.artifacts import (
    atomic_write_json,
    require_complete_manifest,
)
from Uncertainty_Quantification.ConfidenceHead.confidence_head.binning import (
    fixed_linear_binning,
)
from Uncertainty_Quantification.ConfidenceHead.confidence_head.config import (
    BinningConfig,
    ConfidenceConfig,
    load_config,
)
from Uncertainty_Quantification.ConfidenceHead.confidence_head.identity import (
    config_id,
    stable_id,
)


def _valid_config(tmp_path: Path, *, profile: str = "production") -> dict[str, Any]:
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    splits = {}
    for index, name in enumerate(("train", "validation", "test"), start=1):
        split = tmp_path / f"{name}.xyz"
        split.write_text(name, encoding="utf-8")
        splits[name] = {
            "path": str(split),
            "expected_sha256": f"{index:064x}",
        }
    return {
        "profile": profile,
        "checkpoint": {
            "path": str(checkpoint),
            "expected_sha256": "f" * 64,
        },
        "data": splits,
    }


def test_production_rejects_identical_splits(tmp_path: Path) -> None:
    raw = _valid_config(tmp_path)
    config = ConfidenceConfig.model_validate(raw)
    assert (
        len(
            {
                config.data.train.expected_sha256,
                config.data.validation.expected_sha256,
                config.data.test.expected_sha256,
            }
        )
        == 3
    )

    raw["data"]["validation"]["expected_sha256"] = raw["data"]["train"][
        "expected_sha256"
    ]
    with pytest.raises(ValidationError, match="distinct"):
        ConfidenceConfig.model_validate(raw)


def test_smoke_allows_identical_splits_only_when_explicit(tmp_path: Path) -> None:
    raw = _valid_config(tmp_path, profile="smoke")
    repeated_sha = raw["data"]["train"]["expected_sha256"]
    raw["data"]["validation"]["expected_sha256"] = repeated_sha

    with pytest.raises(ValidationError, match="allow_identical_splits"):
        ConfidenceConfig.model_validate(raw)

    raw["allow_identical_splits"] = False
    with pytest.raises(ValidationError, match="allow_identical_splits"):
        ConfidenceConfig.model_validate(raw)

    raw["allow_identical_splits"] = True
    config = ConfidenceConfig.model_validate(raw)
    assert config.profile == "smoke"
    assert config.allow_identical_splits is True


def test_unknown_config_key_is_rejected(tmp_path: Path) -> None:
    raw = _valid_config(tmp_path)
    raw["trainer"] = {"max_epochs": 200, "typo": True}

    with pytest.raises(ValidationError, match="typo"):
        ConfidenceConfig.model_validate(raw)


def test_paths_resolve_from_repo_root_not_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo"
    config_dir = repo_root / "configs"
    config_dir.mkdir(parents=True)
    raw = _valid_config(tmp_path)
    raw["checkpoint"]["path"] = "models/model.ckpt"
    raw["data"]["train"]["path"] = "data/train.xyz"
    raw["data"]["validation"]["path"] = "data/validation.xyz"
    raw["data"]["test"]["path"] = "data/test.xyz"
    raw["run"] = {"output_root": "outputs/confidence"}
    config_path = config_dir / "confidence.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    other_cwd = tmp_path / "elsewhere"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)

    config = load_config(Path("configs/confidence.yaml"), repo_root=repo_root)

    assert config.checkpoint.path == (repo_root / "models/model.ckpt").resolve()
    assert config.data.train.path == (repo_root / "data/train.xyz").resolve()
    assert config.data.validation.path == (repo_root / "data/validation.xyz").resolve()
    assert config.data.test.path == (repo_root / "data/test.xyz").resolve()
    assert config.run.output_root == (repo_root / "outputs/confidence").resolve()


@pytest.mark.parametrize("field", ["force_num_bins", "energy_num_bins"])
def test_binning_rejects_legacy_num_bin_fields(field: str) -> None:
    with pytest.raises(ValidationError, match=field):
        BinningConfig.model_validate({field: 3})


def test_model_heads_are_the_single_num_bins_source(tmp_path: Path) -> None:
    raw = _valid_config(tmp_path)
    raw["model"] = {
        "force": {"num_bins": 7},
        "energy": {"num_bins": 11},
    }

    config = ConfidenceConfig.model_validate(raw)
    force_spec = fixed_linear_binning(
        config.model.force.num_bins,
        config.binning.force_max_error,
    )
    energy_spec = fixed_linear_binning(
        config.model.energy.num_bins,
        config.binning.energy_max_error,
    )

    assert force_spec.num_bins == 7
    assert energy_spec.num_bins == 11


def test_load_config_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    raw = _valid_config(tmp_path, profile="smoke")
    config_path = tmp_path / "duplicate.yaml"
    config_path.write_text(
        f"{yaml.safe_dump(raw)}profile: smoke\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate"):
        load_config(config_path, repo_root=tmp_path)


def test_config_id_accepts_resolved_config_paths(tmp_path: Path) -> None:
    raw = _valid_config(tmp_path)
    config_path = tmp_path / "confidence.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    config = load_config(config_path, repo_root=tmp_path)
    first = config_id(config.model_dump())
    second = config_id(config.model_dump())

    assert first == second
    assert first.startswith("config-")


def test_stable_id_ignores_mapping_order() -> None:
    first = {"outer": {"b": 2, "a": 1}, "items": [3, 4]}
    reordered = {"items": [3, 4], "outer": {"a": 1, "b": 2}}

    first_id = stable_id("config", first)
    reordered_id = stable_id("config", reordered)

    assert first_id == reordered_id
    assert first_id.startswith("config-")
    assert len(first_id) == len("config-") + 16


def test_atomic_json_never_exposes_partial_file(tmp_path: Path) -> None:
    target = tmp_path / "manifest.json"
    original = {"status": "complete", "identity": "old"}
    target.write_text(json.dumps(original), encoding="utf-8")

    with pytest.raises(TypeError):
        atomic_write_json(target, {"status": "complete", "bad": object()})

    assert json.loads(target.read_text(encoding="utf-8")) == original
    assert list(tmp_path.iterdir()) == [target]


def test_incomplete_manifest_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({"status": "incomplete", "identity": "cache-123"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="complete"):
        require_complete_manifest(path, expected_identity="cache-123")


CONFIDENCE_ROOT = Path(__file__).resolve().parents[1]


def test_config_exposes_independent_heads_scheduler_batch_and_logging(
    tmp_path: Path,
) -> None:
    """Protect independently configurable force and energy head workflows."""
    raw = _valid_config(tmp_path)
    raw["model"] = {
        "force": {"enabled": True, "hidden_dims": [8], "dropout": 0.1, "num_bins": 31},
        "energy": {
            "enabled": True,
            "hidden_dims": [12, 6],
            "dropout": 0.2,
            "num_bins": 17,
            "cumulant_order": 4,
            "signed_root": False,
        },
    }
    raw["scheduler"] = {
        "name": "reduce_lr_on_plateau",
        "monitor": "val/total_loss_ema",
        "factor": 0.25,
        "patience": 3,
        "threshold": 1e-5,
        "threshold_mode": "abs",
        "cooldown": 2,
        "min_lr": 1e-7,
    }
    raw["trainer"] = {
        "batch_size": 64,
        "max_epochs": 9,
        "ema_beta": 0.9,
        "early_stopping_patience": 4,
        "min_delta": 1e-5,
        "min_epochs": 2,
        "monitor": "val/total_loss_ema",
        "resume_from": str(tmp_path / "outputs/runs/demo/checkpoints/last.pt"),
    }
    raw["run"] = {"name_prefix": "demo"}
    raw["logging"] = {
        "jsonl": True,
        "wandb": True,
        "wandb_mode": "offline",
        "wandb_project": "upet-confidence-head",
    }

    config = ConfidenceConfig.model_validate(raw)

    assert config.model.force.hidden_dims == (8,)
    assert config.model.energy.hidden_dims == (12, 6)
    assert config.trainer.batch_size == 64
    assert config.scheduler.factor == 0.25
    assert config.logging.wandb_mode == "offline"


@pytest.mark.parametrize(
    ("section", "value", "match"),
    [
        ("run", {"name_prefix": "../unsafe"}, "name_prefix"),
        ("logging", {"wandb_mode": "disabled"}, "wandb_mode"),
        ("model", {"force": {"hidden_dims": []}}, "hidden_dims"),
        ("model", {"energy": {"enabled": False}}, "enabled"),
        ("trainer", {"batch_size": 0}, "batch_size"),
        ("optimizer", {"name": "sgd"}, "name"),
        ("scheduler", {"monitor": "val/loss"}, "monitor"),
        ("trainer", {"monitor": "val/loss"}, "monitor"),
    ],
)
def test_config_rejects_invalid_workflow_settings(
    tmp_path: Path,
    section: str,
    value: dict[str, Any],
    match: str,
) -> None:
    """Protect the immutable workflow constraints from unsafe substitutions."""
    raw = _valid_config(tmp_path)
    raw[section] = value

    with pytest.raises(ValidationError, match=match):
        ConfidenceConfig.model_validate(raw)


@pytest.mark.parametrize(
    ("name", "profile", "device", "wandb_mode"),
    [
        ("n20_local_cpu.yaml", "smoke", "cpu", "offline"),
        ("n20_cpu.yaml", "smoke", "cpu", "offline"),
        ("full_gpu.yaml", "production", "cuda", "online"),
    ],
)
def test_shipped_config_contracts(
    name: str, profile: str, device: str, wandb_mode: str
) -> None:
    """Protect the runnable profile, device, logging, and binning contracts."""
    config = load_config(CONFIDENCE_ROOT / "configs" / name)

    assert config.profile == profile
    assert config.run.device == device
    assert config.logging.wandb_mode == wandb_mode
    assert config.binning.force_max_error == 0.5
    assert config.binning.energy_max_error == 0.3
    assert config.trainer.monitor == "val/total_loss_ema"


@pytest.mark.parametrize(
    (
        "name",
        "hidden_dims",
        "trainer_batch_size",
        "max_epochs",
        "early_stopping_patience",
        "name_prefix",
    ),
    [
        ("n20_local_cpu.yaml", (16,), 2, 1, 1, "upet_n20_local_cpu"),
        ("n20_cpu.yaml", (16,), 2, 1, 1, "upet_n20_cpu"),
        ("full_gpu.yaml", (256, 256, 256), 128, 100, 20, "upet_full"),
    ],
)
def test_shipped_config_training_defaults(
    name: str,
    hidden_dims: tuple[int, ...],
    trainer_batch_size: int,
    max_epochs: int,
    early_stopping_patience: int,
    name_prefix: str,
) -> None:
    """Protect the approved smoke and production compute budgets."""
    config = load_config(CONFIDENCE_ROOT / "configs" / name)

    assert config.model.force.hidden_dims == hidden_dims
    assert config.model.energy.hidden_dims == hidden_dims
    assert config.cache.batch_size == 2
    assert config.trainer.batch_size == trainer_batch_size
    assert config.trainer.max_epochs == max_epochs
    assert config.trainer.early_stopping_patience == early_stopping_patience
    assert config.run.name_prefix == name_prefix


def test_load_config_preserves_absolute_symlink_spelling(tmp_path: Path) -> None:
    physical_dir = tmp_path / "physical"
    physical_dir.mkdir()
    checkpoint = physical_dir / "checkpoint.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    visible_dir = tmp_path / "visible"
    try:
        visible_dir.symlink_to(physical_dir, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("directory symlinks are unavailable on this platform")

    raw = _valid_config(tmp_path)
    configured_checkpoint = visible_dir / checkpoint.name
    raw["checkpoint"]["path"] = str(configured_checkpoint)
    config_path = tmp_path / "confidence.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    config = load_config(config_path, repo_root=tmp_path)

    assert config.checkpoint.path == configured_checkpoint
