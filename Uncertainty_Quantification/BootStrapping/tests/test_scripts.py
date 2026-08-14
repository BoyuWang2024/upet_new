from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from test_config import _write_config


def test_run_stage_returns_two_and_reports_domain_failure(
    tmp_path: Path, capsys
) -> None:
    cli_module = importlib.import_module(
        "Uncertainty_Quantification.BootStrapping.scripts._cli"
    )
    errors_module = importlib.import_module(
        "Uncertainty_Quantification.BootStrapping.bootstrap.errors"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    config_path = _write_config(tmp_path / "config.yaml")

    def fail_stage(config) -> None:
        assert config.training.batch_size == 7
        raise errors_module.HardFailure("stage contract failed")

    status = cli_module.run_stage(
        parser,
        ["--config", str(config_path)],
        fail_stage,
    )

    captured = capsys.readouterr()
    assert status == 2
    assert captured.out == ""
    assert captured.err == "stage contract failed\n"


def test_run_stage_returns_zero_after_success(tmp_path: Path) -> None:
    cli_module = importlib.import_module(
        "Uncertainty_Quantification.BootStrapping.scripts._cli"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    config_path = _write_config(tmp_path / "config.yaml")
    seen: list[int] = []

    def succeed(config) -> None:
        seen.append(config.prediction.batch_size)

    status = cli_module.run_stage(
        parser,
        ["--config", str(config_path)],
        succeed,
    )

    assert status == 0
    assert seen == [5]
