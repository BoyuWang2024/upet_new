from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from Uncertainty_Quantification.LLPR.llpr.config import (
    REPO_ROOT,
    load_llpr_config,
)


ConfigWriter = Callable[[dict[str, Any] | None], Path]


def test_relative_paths_resolve_from_repository_root(
    write_llpr_config: ConfigWriter,
) -> None:
    config = load_llpr_config(write_llpr_config(None))

    assert config.checkpoint.path == (
        REPO_ROOT / "data/checkpoint/pet-omatpes-l-v0.1.0.ckpt"
    )
    assert config.data.build == REPO_ROOT / "data/dataset/matpes_n20.extxyz"
    assert config.output.root == REPO_ROOT / "Uncertainty_Quantification/LLPR/outputs"
    assert config.calibration.ridge.eta is not None
    assert config.calibration.ridge.eta.energy == pytest.approx(1.0e-6)


def test_fixed_scalar_eta_is_normalized_for_both_targets(
    write_llpr_config: ConfigWriter,
) -> None:
    path = write_llpr_config(None)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["calibration"]["ridge"]["eta"] = 2.5e-5
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    config = load_llpr_config(path)

    assert config.calibration.ridge.eta is not None
    assert config.calibration.ridge.eta.energy == pytest.approx(2.5e-5)
    assert config.calibration.ridge.eta.force == pytest.approx(2.5e-5)


def test_fit_mode_rejects_fixed_eta(write_llpr_config: ConfigWriter) -> None:
    path = write_llpr_config(None)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["calibration"]["ridge"] = {
        "mode": "fit",
        "max_condition_number": 1.0e10,
        "eta": {"energy": 1.0e-6, "force": 1.0e-6},
        "fit": {"candidate_multipliers": [1, 3, 10], "score": "gaussian_nll"},
    }
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="fit mode must not define eta"):
        load_llpr_config(path)


def test_fit_mode_requires_fit_settings(write_llpr_config: ConfigWriter) -> None:
    path = write_llpr_config(None)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["calibration"]["ridge"] = {"mode": "fit"}
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="fit mode must define fit"):
        load_llpr_config(path)


def test_unknown_fields_are_rejected(write_llpr_config: ConfigWriter) -> None:
    path = write_llpr_config({"unexpected": True})

    with pytest.raises(ValueError, match="unexpected"):
        load_llpr_config(path)
