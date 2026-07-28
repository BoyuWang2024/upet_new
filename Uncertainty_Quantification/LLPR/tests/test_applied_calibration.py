import json
from pathlib import Path

from Uncertainty_Quantification.LLPR.llpr.inference import (
    AppliedCalibration,
    _load_calibration,
)


def test_load_calibration_accepts_unavailable_legacy_diagnostics(
    tmp_path: Path,
) -> None:
    selected = {}
    for target, alpha in (("energy", 2.0), ("force", 0.5)):
        selected[target] = {
            "target": target,
            "eta": 1.0e-6,
            "alpha": alpha,
            "alpha_sq": alpha**2,
            "gaussian_nll": None,
            "coverage_1sigma": None,
            "coverage_2sigma": None,
            "coverage_3sigma": None,
            "diagnostics_status": "unavailable_from_legacy_validation_summary",
        }
    (tmp_path / "summary.json").write_text(
        json.dumps({"mode": "fixed", "selected": selected}),
        encoding="utf-8",
    )

    calibration = _load_calibration(tmp_path)

    assert calibration == {
        "energy": AppliedCalibration(
            target="energy", eta=1.0e-6, alpha=2.0, alpha_sq=4.0
        ),
        "force": AppliedCalibration(
            target="force", eta=1.0e-6, alpha=0.5, alpha_sq=0.25
        ),
    }
