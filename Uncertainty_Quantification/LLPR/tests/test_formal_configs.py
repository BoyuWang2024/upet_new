from pathlib import Path

import pytest

from Uncertainty_Quantification.LLPR.llpr.config import load_llpr_config
from Uncertainty_Quantification.LLPR.llpr.plot_multi import load_plot_config


CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize(
    ("name", "experiment", "reuse_calibration"),
    [
        ("gpu_mad_test_fixed.yaml", "mad_test", False),
        ("gpu_matpes_train_fixed.yaml", "matpes_train", True),
    ],
)
def test_formal_reuse_config_loads(
    name: str, experiment: str, reuse_calibration: bool
) -> None:
    config = load_llpr_config(CONFIGS / name)

    assert config.output.experiment == experiment
    assert config.data.build is None
    assert config.reuse is not None
    assert config.reuse.curvature is not None
    assert (config.reuse.calibration is not None) is reuse_calibration


def test_formal_three_dataset_plot_config_loads() -> None:
    config = load_plot_config(CONFIGS / "plot_three_datasets.yaml")

    assert tuple(item.label for item in config.evaluations) == (
        "matpes_test",
        "mad_test",
        "matpes_train",
    )
    assert config.output_root.name == "LLPR"
    assert config.style.grid_size == 160
