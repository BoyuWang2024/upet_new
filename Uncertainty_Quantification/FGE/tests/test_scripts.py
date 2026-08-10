"""Contract tests for the five independent formal FGE command-line entry points."""

from __future__ import annotations

import importlib

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

