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
def test_binning_config_matches_fixed_linear_contract(field: str) -> None:
    with pytest.raises(ValidationError, match=field):
        BinningConfig.model_validate({field: 2})

    config = BinningConfig(force_num_bins=3, energy_num_bins=3)
    force_spec = fixed_linear_binning(
        config.force_num_bins,
        config.force_max_error,
    )
    energy_spec = fixed_linear_binning(
        config.energy_num_bins,
        config.energy_max_error,
    )

    assert force_spec.num_bins == 3
    assert energy_spec.num_bins == 3


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
