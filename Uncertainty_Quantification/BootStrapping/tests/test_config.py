from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def _write_config(path: Path, *, extra_root: str = "") -> Path:
    path.write_text(
        f"""
schema_version: 1
experiment:
  name: bootstrap-test
  run_id: run-a
  output_root: outputs
checkpoint:
  base_path: checkpoint.pt
data:
  train: train.extxyz
  val: val.extxyz
  test: test.extxyz
  units:
    energy: eV
    forces: eV/Angstrom
    stress: eV/Angstrom^3
bootstrap:
  ensemble_size: 2
  base_seed: 2026
  sample_size: null
  replacement: true
  save_indices: true
  save_oob: true
training:
  batch_size: 7
  max_epochs: 1
  trainable_head: pet_last_layers
  optimizer:
    name: Adam
    learning_rate: 0.00003
    weight_decay: 0.0
  scheduler: none
  gradient_clip: 1.0
  ema_decay: 0.999
  device: cpu
  precision: float32
  num_workers: 0
prediction:
  splits: [val, test]
  parameter_modes: [raw]
  batch_size: 5
  structure_chunk_size: 3
  device: cpu
  num_workers: 0
uncertainty:
  parameter_modes: [raw]
  ddof: 1
  compute_std: true
  compute_gmd: true
  gmd_pairs: distinct_unordered
{extra_root}
""".lstrip(),
        encoding="utf-8",
    )
    return path


def test_load_config_resolves_paths_and_keeps_batches_configurable(
    tmp_path: Path,
) -> None:
    config_module = importlib.import_module(
        "Uncertainty_Quantification.BootStrapping.bootstrap.config"
    )
    source = _write_config(tmp_path / "config.yaml")

    config = config_module.load_config(source)

    assert config.training.batch_size == 7
    assert config.prediction.batch_size == 5
    assert config.data.train == (tmp_path / "train.extxyz").resolve()
    assert config.experiment.output_root == (tmp_path / "outputs").resolve()


def test_load_config_rejects_unknown_root_key(tmp_path: Path) -> None:
    config_module = importlib.import_module(
        "Uncertainty_Quantification.BootStrapping.bootstrap.config"
    )
    errors_module = importlib.import_module(
        "Uncertainty_Quantification.BootStrapping.bootstrap.errors"
    )
    source = _write_config(tmp_path / "config.yaml", extra_root="unexpected: true")

    with pytest.raises(errors_module.HardFailure, match="unknown key.*unexpected"):
        config_module.load_config(source)


def test_load_config_rejects_nonpositive_batch_size(tmp_path: Path) -> None:
    config_module = importlib.import_module(
        "Uncertainty_Quantification.BootStrapping.bootstrap.config"
    )
    errors_module = importlib.import_module(
        "Uncertainty_Quantification.BootStrapping.bootstrap.errors"
    )
    source = _write_config(tmp_path / "config.yaml")
    document = source.read_text(encoding="utf-8").replace(
        "batch_size: 7", "batch_size: 0", 1
    )
    source.write_text(document, encoding="utf-8")

    with pytest.raises(
        errors_module.HardFailure, match="training.batch_size must be at least 1"
    ):
        config_module.load_config(source)
