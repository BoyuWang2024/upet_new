import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from Uncertainty_Quantification.LLPR.llpr import curvature as curvature_module
from Uncertainty_Quantification.LLPR.llpr.config import load_llpr_config
from Uncertainty_Quantification.LLPR.llpr.curvature import run_build


CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.llpr_n20
@pytest.mark.skipif(
    not bool(os.environ.get("UPET_RUN_LLPR_N20")),
    reason="set UPET_RUN_LLPR_N20=1 to run real n20 curvature recovery",
)
def test_n20_interrupted_curvature_matches_uninterrupted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    base = load_llpr_config(CONFIGS / "cpu_n20_fixed.yaml")
    clean_output = base.output.model_copy(
        update={"root": tmp_path / "outputs", "experiment": "curvature_clean"}
    )
    resume_output = base.output.model_copy(
        update={"root": tmp_path / "outputs", "experiment": "curvature_resume"}
    )
    clean_config = base.model_copy(update={"output": clean_output})
    resume_config = base.model_copy(update={"output": resume_output})

    clean_curvature = run_build(clean_config)
    original_compute = curvature_module.compute_structure_jacobians
    call_count = 0

    def interrupt_after_checkpoint(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        if call_count == 6:
            raise RuntimeError("injected curvature interruption")
        return original_compute(*args, **kwargs)

    with monkeypatch.context() as patcher:
        patcher.setattr(
            curvature_module,
            "compute_structure_jacobians",
            interrupt_after_checkpoint,
        )
        with pytest.raises(RuntimeError, match="injected curvature interruption"):
            run_build(resume_config)

    progress_files = list(
        (resume_output.root / resume_output.experiment / "curvature").glob(
            "*/progress.npz"
        )
    )
    assert len(progress_files) == 1
    progress_path = progress_files[0]

    resumed_curvature = run_build(resume_config)
    assert not progress_path.exists()
    with (
        np.load(clean_curvature / "curvature.npz", allow_pickle=False) as clean,
        np.load(resumed_curvature / "curvature.npz", allow_pickle=False) as resumed,
    ):
        np.testing.assert_array_equal(resumed["energy"], clean["energy"])
        np.testing.assert_array_equal(resumed["force"], clean["force"])
    assert json.loads(
        (resumed_curvature / "diagnostics.json").read_text()
    ) == json.loads((clean_curvature / "diagnostics.json").read_text())
