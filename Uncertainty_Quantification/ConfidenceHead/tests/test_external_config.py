from __future__ import annotations

from pathlib import Path

import pytest
from confidence_head.external_config import (
    CacheSplitSource,
    ExistingEvaluationSource,
    ExtXYZSource,
    load_external_config,
)


SHA_TEST = "1" * 64
SHA_TRAIN = "2" * 64
SHA_MAD = "3" * 64
SHA_CHECKPOINT = "4" * 64


def _write_config(tmp_path: Path, mad_source: str | None = None) -> Path:
    mad = (
        mad_source
        or f"""
    source: extxyz
    path: data/mad-test.xyz
    expected_sha256: "{SHA_MAD}"
"""
    )
    path = tmp_path / "external.yaml"
    path.write_text(
        f"""
profile: production
checkpoint:
  path: data/model.ckpt
  expected_sha256: "{SHA_CHECKPOINT}"
datasets:
  matpes_test:
    source: existing_evaluation
    expected_sha256: "{SHA_TEST}"
  matpes_train:
    source: cache_split
    split: train
    expected_sha256: "{SHA_TRAIN}"
  mad_test:
{mad}
runs_root: outputs/runs
cache_root: outputs/prediction_cache
plots_root: plots/confidence_head
batch_size: 16
device: cuda
""",
        encoding="utf-8",
    )
    return path


def test_external_config_loads_three_mutually_exclusive_sources(
    tmp_path: Path,
) -> None:
    config = load_external_config(_write_config(tmp_path), repo_root=tmp_path)

    assert isinstance(config.datasets["matpes_test"], ExistingEvaluationSource)
    assert isinstance(config.datasets["matpes_train"], CacheSplitSource)
    assert isinstance(config.datasets["mad_test"], ExtXYZSource)
    assert config.datasets["matpes_test"].source == "existing_evaluation"
    assert config.datasets["matpes_train"].source == "cache_split"
    assert config.datasets["mad_test"].source == "extxyz"
    assert config.datasets["mad_test"].path == tmp_path / "data/mad-test.xyz"
    assert config.runs_root == tmp_path / "outputs/runs"


def test_external_config_rejects_fields_from_another_source(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        mad_source=f"""
    source: extxyz
    path: data/mad-test.xyz
    split: train
    expected_sha256: "{SHA_MAD}"
""",
    )

    with pytest.raises(ValueError, match="split|Extra inputs"):
        load_external_config(path, repo_root=tmp_path)


def test_production_requires_exact_dataset_names(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    text = path.read_text(encoding="utf-8").replace("mad_test:", "other:")
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="matpes_test.*matpes_train.*mad_test"):
        load_external_config(path, repo_root=tmp_path)


def test_external_config_rejects_unsafe_dataset_name(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    text = path.read_text(encoding="utf-8").replace("mad_test:", "../mad_test:")
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="safe path component"):
        load_external_config(path, repo_root=tmp_path)


def test_smoke_allows_a_selected_subset(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    text = path.read_text(encoding="utf-8").replace(
        "profile: production", "profile: smoke"
    )
    start = text.index("  matpes_test:")
    end = text.index("runs_root:")
    smoke = f"""  smoke:
    source: extxyz
    path: data/n20.extxyz
    expected_sha256: "{SHA_MAD}"
"""
    path.write_text(text[:start] + smoke + text[end:], encoding="utf-8")

    config = load_external_config(path, repo_root=tmp_path)

    assert set(config.datasets) == {"smoke"}
