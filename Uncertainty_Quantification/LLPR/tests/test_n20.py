import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from Uncertainty_Quantification.LLPR.llpr.artifacts import verify_run
from Uncertainty_Quantification.LLPR.llpr.calibration import run_calibrate
from Uncertainty_Quantification.LLPR.llpr.config import load_llpr_config
from Uncertainty_Quantification.LLPR.llpr.curvature import run_build
from Uncertainty_Quantification.LLPR.llpr.inference import run_evaluate
from Uncertainty_Quantification.LLPR.llpr.plotting import PlotConfig, run_plot


CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _manifest(path: Path) -> dict[str, object]:
    return json.loads((path / "manifest.json").read_text(encoding="utf-8"))


def _summary(path: Path) -> dict[str, Any]:
    return json.loads((path / "summary.json").read_text(encoding="utf-8"))


def _assert_positive_finite_variances(evaluation: Path) -> None:
    with np.load(evaluation / "details.npz", allow_pickle=False) as details:
        assert len(details["structure_index"]) == 20
        assert len(details["energy_calibrated_var"]) == 20
        for name in (
            "energy_raw_var",
            "energy_calibrated_var",
            "energy_calibrated_std",
            "force_raw_var_component",
            "force_calibrated_var_component",
            "force_calibrated_std_component",
        ):
            values = details[name]
            assert np.all(np.isfinite(values)), name
            assert np.all(values > 0), name
        assert len(details["force_residual"]) == int(details["force_offsets"][-1])


@pytest.mark.llpr_n20
@pytest.mark.skipif(
    not bool(os.environ.get("UPET_RUN_LLPR_N20")),
    reason="set UPET_RUN_LLPR_N20=1 to run the real n20 full path",
)
def test_n20_fixed_and_fit_full_paths() -> None:
    fixed = load_llpr_config(CONFIGS / "cpu_n20_fixed.yaml")
    fitted = load_llpr_config(CONFIGS / "cpu_n20_fit.yaml")

    fixed_curvature = run_build(fixed)
    fixed_calibration = run_calibrate(fixed)
    fixed_evaluation = run_evaluate(fixed)
    fitted_curvature = run_build(fitted)
    fitted_calibration = run_calibrate(fitted)
    fitted_evaluation = run_evaluate(fitted)

    run_root = fixed.output.root / fixed.output.experiment
    assert run_root == fitted.output.root / fitted.output.experiment
    assert fixed_curvature == fitted_curvature
    assert (
        _manifest(fixed_calibration)["curvature_identity"]
        == _manifest(fitted_calibration)["curvature_identity"]
    )

    fixed_selected = _summary(fixed_calibration)["selected"]
    assert fixed_selected["energy"]["eta"] == 1.0e-6
    assert fixed_selected["force"]["eta"] == 1.0e-6
    fitted_selected = _summary(fitted_calibration)["selected"]
    assert fitted_selected["energy"]["eta"] > 0
    assert fitted_selected["force"]["eta"] > 0
    assert np.isfinite(fitted_selected["energy"]["alpha"])
    assert np.isfinite(fitted_selected["force"]["alpha"])

    fitted_candidates = json.loads(
        (fitted_calibration / "candidates.json").read_text(encoding="utf-8")
    )
    assert len(fitted_candidates["energy"]) > 1
    assert len(fitted_candidates["force"]) > 1

    _assert_positive_finite_variances(fixed_evaluation)
    _assert_positive_finite_variances(fitted_evaluation)
    for evaluation in (fixed_evaluation, fitted_evaluation):
        plot = run_plot(
            PlotConfig(
                run_root=run_root,
                evaluation_identity=str(_manifest(evaluation)["identity"]),
                bin_count=5,
                sample_size=10_000,
                seed=2026,
            )
        )
        assert (plot / "manifest.json").is_file()

    result = verify_run(run_root, level="full")
    assert result["status"] == "complete"
