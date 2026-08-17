from __future__ import annotations

import importlib
import tarfile
from pathlib import Path

import pytest

from Uncertainty_Quantification.BootStrapping.bootstrap.release import (
    build_source_release,
)


BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("module_name", "message"),
    [
        ("predict_campaign", "campaign prediction failed:"),
        ("compute_campaign_uq", "campaign UQ failed:"),
        ("plot_campaign", "campaign plotting failed:"),
    ],
)
def test_campaign_entrypoints_report_hard_failures(
    tmp_path: Path, capsys, module_name: str, message: str
) -> None:
    module = importlib.import_module(
        f"Uncertainty_Quantification.BootStrapping.scripts.{module_name}"
    )

    status = module.main(["--campaign", str(tmp_path / "missing.yaml")])

    captured = capsys.readouterr()
    assert status == 2
    assert captured.out == ""
    assert captured.err.startswith(message)


def test_formal_release_contains_campaign_commands_and_excludes_plots(
    tmp_path: Path,
) -> None:
    archive = build_source_release(
        BOOTSTRAP_ROOT, tmp_path / "bootstrap-release.tar.gz"
    )

    with tarfile.open(archive, "r:gz") as handle:
        names = set(handle.getnames())
    for relative in (
        "bootstrap/campaign.py",
        "bootstrap/native_prediction.py",
        "bootstrap/uq_campaign.py",
        "bootstrap/plot_store.py",
        "scripts/predict_campaign.py",
        "scripts/compute_campaign_uq.py",
        "scripts/plot_campaign.py",
        "configs/three_run_three_dataset_raw.yaml",
        "README.md",
    ):
        assert f"BootStrapping/{relative}" in names
    assert all("/tests/" not in name for name in names)
    assert all("/outputs/" not in name for name in names)
    assert all("internal_migration" not in name for name in names)
    assert all("/Plots/" not in name for name in names)
    assert all("ConfidenceHead" not in name for name in names)


def test_readme_documents_formal_campaign_commands() -> None:
    readme = (BOOTSTRAP_ROOT / "README.md").read_text(encoding="utf-8")

    assert "scripts.predict_campaign" in readme
    assert "scripts.compute_campaign_uq" in readme
    assert "scripts.plot_campaign" in readme
