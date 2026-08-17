from __future__ import annotations

from pathlib import Path

from Uncertainty_Quantification.BootStrapping.bootstrap.campaign import load_campaign


BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]


def test_formal_campaign_has_three_runs_and_three_datasets() -> None:
    campaign = load_campaign(
        BOOTSTRAP_ROOT / "configs" / "three_run_three_dataset_raw.yaml"
    )

    assert [run.label for run in campaign.runs] == [
        "full_remote_b8_e8",
        "lr_1e-4",
        "lr_1e-6",
    ]
    assert [dataset.label for dataset in campaign.datasets] == [
        "matpes_test",
        "mad_test",
        "matpes_train",
    ]
    assert [dataset.reference_targets for dataset in campaign.datasets] == [
        ("energy", "forces", "stress"),
        ("energy", "forces"),
        ("energy", "forces", "stress"),
    ]
    assert campaign.prediction.mode == "raw"
    assert campaign.prediction.member_count == 8
    assert campaign.prediction.device == "cuda"


def test_formal_campaign_dataset_paths_are_configuration_driven() -> None:
    campaign = load_campaign(
        BOOTSTRAP_ROOT / "configs" / "three_run_three_dataset_raw.yaml"
    )

    assert [dataset.path.name for dataset in campaign.datasets] == [
        "matpes_test.extxyz",
        "mad-test.xyz",
        "matpes_train.extxyz",
    ]
