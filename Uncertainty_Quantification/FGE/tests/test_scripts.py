"""Contract tests for the five independent formal FGE command-line entry points."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "name",
    ("preflight", "train", "predict", "evaluate", "validate"),
)
def test_each_formal_cli_exposes_one_required_config_option(name: str) -> None:
    """Every formal stage must expose a parser with exactly one --config option."""
    module = importlib.import_module(f"Uncertainty_Quantification.FGE.scripts.{name}")
    parser = module.build_parser()

    assert [action.dest for action in parser._actions].count("config") == 1


@pytest.mark.parametrize(
    "name",
    ("preflight", "train", "predict", "evaluate", "validate"),
)
def test_each_formal_cli_supports_direct_script_execution(name: str) -> None:
    """The documented direct script form must be executable."""
    script = Path(__file__).parents[1] / "scripts" / f"{name}.py"

    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout


@pytest.mark.parametrize("stage", ("run-all", "uq", "plot", "unknown"))
def test_preflight_cli_rejects_nonformal_stage_names(stage: str) -> None:
    """Only the three formal stages are accepted; legacy umbrella names are errors."""
    module = importlib.import_module("Uncertainty_Quantification.FGE.scripts.preflight")

    with pytest.raises(SystemExit) as failure:
        module.build_parser().parse_args(["--config", "fge.yaml", "--stage", stage])

    assert failure.value.code == 2


def test_cli_returns_nonzero_for_a_hard_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A formal stage reports its hard failure and never exits successfully."""
    from Uncertainty_Quantification.FGE.scripts import _cli

    parser = _cli.build_parser("test")
    monkeypatch.setattr(_cli, "load_config", lambda _: object())

    def fail(_: object) -> object:
        from Uncertainty_Quantification.FGE.fge.errors import HardFailure

        raise HardFailure("intentional failure")

    assert _cli.run_stage(parser, ["--config", "fge.yaml"], stage=fail) == 2
    assert "intentional failure" in capsys.readouterr().err
