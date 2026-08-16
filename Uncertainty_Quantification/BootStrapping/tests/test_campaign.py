from __future__ import annotations

from pathlib import Path

import pytest

from Uncertainty_Quantification.BootStrapping.bootstrap.campaign import (
    load_campaign,
    select_campaign_items,
)
from Uncertainty_Quantification.BootStrapping.bootstrap.errors import HardFailure


def _write_run_config(path: Path, run_id: str) -> None:
    path.write_text(
        f"""
schema_version: 1
experiment:
  name: bootstrap-test
  run_id: {run_id}
  output_root: outputs
checkpoint:
  base_path: checkpoint.pt
data:
  train: train.extxyz
  val: val.extxyz
  test: test.extxyz
  units: {{energy: eV, forces: eV/Angstrom, stress: eV/Angstrom^3}}
bootstrap:
  ensemble_size: 8
  base_seed: 2026
  sample_size: null
  replacement: true
  save_indices: true
  save_oob: true
training:
  batch_size: 7
  max_epochs: 1
  trainable_head: pet_last_layers
  optimizer: {{name: Adam, learning_rate: 0.00003, weight_decay: 0.0}}
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
""".lstrip(),
        encoding="utf-8",
    )


def _write_campaign(
    tmp_path: Path,
    *,
    dataset_storage_key: str = "mad_test",
    duplicate_run: bool = False,
    mode: str = "raw",
) -> Path:
    run_labels = ["full_remote_b8_e8", "lr_1e-4", "lr_1e-6"]
    for label in run_labels:
        _write_run_config(tmp_path / f"{label}.yaml", label)

    repeated_label = "full_remote_b8_e8" if duplicate_run else "lr_1e-4"
    source = tmp_path / "campaign.yaml"
    source.write_text(
        f"""
schema_version: 1
runs:
  - label: full_remote_b8_e8
    config: full_remote_b8_e8.yaml
    root: runs/full_remote_b8_e8
  - label: {repeated_label}
    config: lr_1e-4.yaml
    root: runs/lr_1e-4
  - label: lr_1e-6
    config: lr_1e-6.yaml
    root: runs/lr_1e-6
datasets:
  - label: matpes_test
    storage_key: test
    path: data/matpes_test.extxyz
    reference_targets: [energy, forces, stress]
  - label: mad_test
    storage_key: {dataset_storage_key}
    path: data/mad-test.xyz
    reference_targets: [energy, forces]
  - label: matpes_train
    storage_key: matpes_train
    path: data/matpes_train.extxyz
    reference_targets: [energy, forces, stress]
prediction:
  mode: {mode}
  member_count: 8
  device: cpu
  batch_size: 4
plot:
  grid_size: 160
  gaussian_sigma: 1.2
  contour_masses: [0.5, 0.7, 0.85, 0.95, 0.99]
  scatter_max_points: 20000
  scatter_seed: 20260714
  scatter_size: 4.0
  scatter_alpha: 0.7
  log_margin: 0.05
  figure_size: [7.0, 7.0]
  dpi: 300
  formats: [png, pdf]
output_root: plots
""".lstrip(),
        encoding="utf-8",
    )
    return source


def test_campaign_resolves_runs_datasets_and_shared_output(tmp_path: Path) -> None:
    campaign = load_campaign(_write_campaign(tmp_path))

    assert [run.label for run in campaign.runs] == [
        "full_remote_b8_e8",
        "lr_1e-4",
        "lr_1e-6",
    ]
    assert [
        (item.label, item.storage_key, item.reference_targets)
        for item in campaign.datasets
    ] == [
        ("matpes_test", "test", ("energy", "forces", "stress")),
        ("mad_test", "mad_test", ("energy", "forces")),
        ("matpes_train", "matpes_train", ("energy", "forces", "stress")),
    ]
    assert campaign.prediction.mode == "raw"
    assert campaign.prediction.member_count == 8
    assert campaign.output_root == (tmp_path / "plots").resolve()
    assert campaign.runs[0].config_path == (tmp_path / "full_remote_b8_e8.yaml").resolve()
    assert campaign.runs[0].run_root == (tmp_path / "runs/full_remote_b8_e8").resolve()
    assert campaign.datasets[1].path == (tmp_path / "data/mad-test.xyz").resolve()


@pytest.mark.parametrize("value", ["../escape", "Mad-Test", "a/b", ".", ""])
def test_campaign_rejects_unsafe_artifact_keys(tmp_path: Path, value: str) -> None:
    source = _write_campaign(tmp_path, dataset_storage_key=value)

    with pytest.raises(HardFailure, match="storage_key"):
        load_campaign(source)


def test_campaign_rejects_duplicate_labels_and_ema(tmp_path: Path) -> None:
    with pytest.raises(HardFailure, match="duplicate run label"):
        load_campaign(_write_campaign(tmp_path, duplicate_run=True))
    with pytest.raises(HardFailure, match="mode must be raw"):
        load_campaign(_write_campaign(tmp_path, mode="ema"))


def test_campaign_rejects_mismatched_run_ensemble_size(tmp_path: Path) -> None:
    source = _write_campaign(tmp_path)
    run_config = tmp_path / "lr_1e-6.yaml"
    run_config.write_text(
        run_config.read_text(encoding="utf-8").replace(
            "ensemble_size: 8", "ensemble_size: 2"
        ),
        encoding="utf-8",
    )

    with pytest.raises(HardFailure, match="ensemble size"):
        load_campaign(source)


def test_campaign_rejects_invalid_plot_contract(tmp_path: Path) -> None:
    source = _write_campaign(tmp_path)
    source.write_text(
        source.read_text(encoding="utf-8").replace(
            "contour_masses: [0.5, 0.7, 0.85, 0.95, 0.99]",
            "contour_masses: [0.5, 0.5]",
        ),
        encoding="utf-8",
    )

    with pytest.raises(HardFailure, match="contour_masses"):
        load_campaign(source)


def test_select_campaign_items_preserves_campaign_order(tmp_path: Path) -> None:
    campaign = load_campaign(_write_campaign(tmp_path))

    runs, datasets = select_campaign_items(
        campaign,
        run_labels=("lr_1e-6", "full_remote_b8_e8"),
        dataset_labels=("matpes_train", "mad_test"),
    )

    assert [run.label for run in runs] == ["full_remote_b8_e8", "lr_1e-6"]
    assert [dataset.label for dataset in datasets] == ["mad_test", "matpes_train"]
