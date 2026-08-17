from pathlib import Path

from Uncertainty_Quantification.BootStrapping.bootstrap.campaign import load_campaign


def test_formal_campaign_reuses_migrated_matpes_test_storage() -> None:
    root = Path(__file__).resolve().parents[1]
    campaign = load_campaign(root / "configs" / "three_run_three_dataset_raw.yaml")

    assert [dataset.storage_key for dataset in campaign.datasets] == [
        "test",
        "mad_test",
        "matpes_train",
    ]
