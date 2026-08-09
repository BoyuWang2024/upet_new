from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from Uncertainty_Quantification.FGE.fge.config import load_config
from Uncertainty_Quantification.FGE.fge.errors import HardFailure

from .conftest import SHA_BASE, SHA_TEST, SHA_TRAIN


def write_config(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_load_config_parses_complete_literal_contract(
    tmp_path: Path, config_payload: dict[str, object]
) -> None:
    config = load_config(write_config(tmp_path / "fge.yaml", config_payload))

    assert config.project.name == "upet_fge_full"
    assert config.training.expected_readout_parameter_count == 13338
    assert config.fge.member_count == 8


@pytest.mark.parametrize(
    ("section", "key"),
    [(None, "unexpected"), ("training", "unexpected"), ("scientific", "unexpected")],
)
def test_load_config_rejects_unknown_keys_recursively(
    tmp_path: Path,
    copy_config,
    section: str | None,
    key: str,
) -> None:
    payload = copy_config()
    target = payload if section is None else payload[section]
    assert isinstance(target, dict)
    target[key] = "not allowed"

    with pytest.raises(HardFailure, match="unknown key"):
        load_config(write_config(tmp_path / "fge.yaml", payload))


def test_load_config_resolves_relative_paths_from_yaml_directory(
    tmp_path: Path, config_payload: dict[str, object]
) -> None:
    config_path = write_config(tmp_path / "nested" / "fge.yaml", config_payload)

    config = load_config(config_path)

    assert (
        config.paths.base_checkpoint == (tmp_path / "nested/models/base.ckpt").resolve()
    )
    assert config.paths.output_root == (tmp_path / "nested/outputs").resolve()


def test_load_config_expands_only_named_runtime_environment_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_payload: dict[str, object]
) -> None:
    names = {
        "UPET_FGE_BASE_CHECKPOINT": "/runtime/base.ckpt",
        "UPET_FGE_TRAIN_DATA": "/runtime/train.extxyz",
        "UPET_FGE_VAL_DATA": "/runtime/val.extxyz",
        "UPET_FGE_TEST_DATA": "/runtime/test.extxyz",
        "UPET_FGE_OUTPUT_ROOT": "/runtime/outputs",
    }
    for name, value in names.items():
        monkeypatch.setenv(name, value)
    paths = config_payload["paths"]
    assert isinstance(paths, dict)
    paths.update(
        {
            "base_checkpoint": "${UPET_FGE_BASE_CHECKPOINT}",
            "train_data": "${UPET_FGE_TRAIN_DATA}",
            "val_data": "${UPET_FGE_VAL_DATA}",
            "test_data": "${UPET_FGE_TEST_DATA}",
            "output_root": "${UPET_FGE_OUTPUT_ROOT}",
        }
    )

    config = load_config(write_config(tmp_path / "fge.yaml", config_payload))

    assert config.paths.base_checkpoint == Path(names["UPET_FGE_BASE_CHECKPOINT"])
    assert config.paths.output_root == Path(names["UPET_FGE_OUTPUT_ROOT"])


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("training", "seed", True),
        ("training", "batch_size", "16"),
        ("training", "drop_last", 1),
        ("paths", "train_data", 1),
        ("evaluation", "structure_quantile", "0.95"),
    ],
)
def test_load_config_rejects_wrong_scalar_types(
    tmp_path: Path, copy_config, section: str, key: str, value: object
) -> None:
    payload = copy_config()
    target = payload[section]
    assert isinstance(target, dict)
    target[key] = value

    with pytest.raises(HardFailure):
        load_config(write_config(tmp_path / "fge.yaml", payload))


def test_load_config_rejects_missing_or_unsupported_environment_variable(
    tmp_path: Path, config_payload: dict[str, object]
) -> None:
    paths = config_payload["paths"]
    assert isinstance(paths, dict)
    paths["train_data"] = "${UPET_FGE_UNSUPPORTED}"

    with pytest.raises(HardFailure, match="environment"):
        load_config(write_config(tmp_path / "fge.yaml", config_payload))


def test_full_and_n20_configs_match_their_scientific_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "UPET_FGE_BASE_CHECKPOINT",
        "UPET_FGE_TRAIN_DATA",
        "UPET_FGE_VAL_DATA",
        "UPET_FGE_TEST_DATA",
        "UPET_FGE_OUTPUT_ROOT",
    ):
        monkeypatch.setenv(name, f"/runtime/{name.lower()}")
    root = Path(__file__).parents[1]
    full = load_config(root / "configs/upet_fge_full.yaml")
    n20 = load_config(root / "configs/upet_fge_n20_cpu.yaml")

    assert (full.fge.member_count, full.fge.cycles, full.fge.epochs_per_cycle) == (
        8,
        8,
        8,
    )
    assert full.training.device == "cuda"
    assert full.scientific.evaluation.scientific_evaluation is True
    assert (n20.fge.member_count, n20.fge.cycles, n20.fge.epochs_per_cycle) == (2, 2, 2)
    assert n20.training.device == "cpu"
    assert n20.scientific.training.path_feasibility_only is True
    assert n20.scientific.evaluation.scientific_evaluation is False


def test_sanitized_config_contains_logical_path_roles_and_no_runtime_paths(
    tmp_path: Path, config_payload: dict[str, object]
) -> None:
    config = load_config(write_config(tmp_path / "fge.yaml", config_payload))

    sanitized = config.sanitized()

    assert sanitized["paths"] == {
        "base_checkpoint": {"role": "base_checkpoint", "sha256": SHA_BASE},
        "train_data": {"role": "train_data", "sha256": SHA_TRAIN},
        "val_data": {"role": "val_data", "sha256": SHA_TRAIN},
        "test_data": {"role": "test_data", "sha256": SHA_TEST},
        "output_root": {"role": "output_root"},
    }
    assert str(tmp_path) not in str(sanitized)
    assert "models/base.ckpt" not in str(sanitized)


@pytest.mark.parametrize("stage", ["train", "predict", "evaluate"])
def test_assert_stage_accepts_the_supported_stages(
    tmp_path: Path, config_payload: dict[str, object], stage: str
) -> None:
    load_config(write_config(tmp_path / "fge.yaml", config_payload)).assert_stage(stage)


def test_assert_stage_rejects_unknown_stage(
    tmp_path: Path, config_payload: dict[str, object]
) -> None:
    with pytest.raises(HardFailure, match="stage"):
        load_config(write_config(tmp_path / "fge.yaml", config_payload)).assert_stage(
            "preflight"
        )
