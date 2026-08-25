from __future__ import annotations

from pathlib import Path

import pytest
from confidence_head.e0_config import load_e0_config
from confidence_head.external_config import ExtXYZSource, load_external_config


SHA = "1" * 64


def _write_e0_config(tmp_path: Path) -> Path:
    path = tmp_path / "e0.yaml"
    path.write_text(
        """
external_config: prediction.yaml
validation_dataset: mad_r2scan_val
test_dataset: mad_r2scan_test
validation_expected:
  structures: 16098
  atoms: 310432
  elements: 89
test_expected:
  structures: 16072
  atoms: 311657
  elements: 89
output_root: outputs/e0
plots_root: plots/e0
plot:
  grid_size: 80
""".lstrip(),
        encoding="utf-8",
    )
    return path


def _write_prediction_config(
    tmp_path: Path, *, test_name: str = "mad_r2scan_test"
) -> Path:
    path = tmp_path / "prediction.yaml"
    path.write_text(
        f"""
profile: postprocessing
checkpoint:
  path: model.ckpt
  expected_sha256: "{SHA}"
datasets:
  mad_r2scan_val:
    source: extxyz
    path: val.extxyz
    expected_sha256: "{SHA}"
  {test_name}:
    source: extxyz
    path: test.extxyz
    expected_sha256: "{SHA}"
runs_root: outputs/runs
cache_root: outputs/cache
plots_root: plots
device: cpu
""".lstrip(),
        encoding="utf-8",
    )
    return path


def test_e0_config_resolves_paths_and_counts(tmp_path: Path) -> None:
    config = load_e0_config(_write_e0_config(tmp_path))

    assert config.external_config == (tmp_path / "prediction.yaml").absolute()
    assert config.output_root == (tmp_path / "outputs/e0").absolute()
    assert config.plots_root == (tmp_path / "plots/e0").absolute()
    assert config.validation_dataset == "mad_r2scan_val"
    assert config.test_dataset == "mad_r2scan_test"
    assert config.validation_expected.structures == 16098
    assert config.test_expected.atoms == 311657
    assert config.plot.grid_size == 80


def test_e0_config_rejects_same_dataset_and_unknown_field(tmp_path: Path) -> None:
    path = _write_e0_config(tmp_path)
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "test_dataset: mad_r2scan_test",
            "test_dataset: mad_r2scan_val",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must differ"):
        load_e0_config(path)

    path = _write_e0_config(tmp_path)
    path.write_text(path.read_text(encoding="utf-8") + "unknown: true\n")
    with pytest.raises(ValueError):
        load_e0_config(path)


def test_postprocessing_profile_requires_exact_val_test_extxyz(
    tmp_path: Path,
) -> None:
    config = load_external_config(
        _write_prediction_config(tmp_path), repo_root=tmp_path
    )

    assert config.profile == "postprocessing"
    assert set(config.datasets) == {"mad_r2scan_val", "mad_r2scan_test"}
    assert all(isinstance(source, ExtXYZSource) for source in config.datasets.values())

    with pytest.raises(ValueError, match="mad_r2scan_val.*mad_r2scan_test"):
        load_external_config(
            _write_prediction_config(tmp_path, test_name="wrong"),
            repo_root=tmp_path,
        )
