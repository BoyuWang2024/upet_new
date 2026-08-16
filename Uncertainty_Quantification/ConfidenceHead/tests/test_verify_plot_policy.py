from __future__ import annotations

from pathlib import Path

import pytest
from confidence_head.workflows import verify


def _stub_complete_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    root = tmp_path / "runs" / "complete"
    root.mkdir(parents=True)
    manifest = {
        "schema_version": verify.RUN_SCHEMA_VERSION,
        "status": "complete",
        "identity": "run-id",
        "run_id": "run-id",
        "artifacts": {},
    }
    monkeypatch.setattr(verify, "_mapping", lambda path: manifest)
    monkeypatch.setattr(verify, "_verify_artifacts", lambda *args, **kwargs: {})
    monkeypatch.setattr(verify, "_verify_identity", lambda *args: "atom_mean")
    return root


def test_verify_run_can_allow_only_derived_plot_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _stub_complete_run(tmp_path, monkeypatch)
    plot = root / "plots" / "argmax_bin_boxplots" / "result.png"
    plot.parent.mkdir(parents=True)
    plot.write_bytes(b"plot")

    with pytest.raises(ValueError, match="forbidden figure"):
        verify.verify_run(root, full=False)

    assert verify.verify_run(root, full=False, allow_plots=True)["status"] == "complete"

    (root / "unexpected.pdf").write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="unexpected.pdf"):
        verify.verify_run(root, full=False, allow_plots=True)
