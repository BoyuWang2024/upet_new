"""Readable, deterministic names and locations for confidence-head runs."""

from __future__ import annotations

import re
from decimal import Decimal
from pathlib import Path

from .config import ConfidenceConfig


_SAFE_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _number(value: float) -> str:
    """Render a configuration float without insignificant trailing zeroes."""
    rendered = format(Decimal(str(value)), "f").rstrip("0").rstrip(".")
    return rendered or "0"


def _hidden_dims(dimensions: tuple[int, ...]) -> str:
    return "x".join(str(dimension) for dimension in dimensions)


def build_run_name(config: ConfidenceConfig) -> str:
    """Build a filesystem-safe name that captures both confidence branches."""
    algorithm = config.binning.algorithm.removesuffix("_v1").replace("_", "-")
    name = (
        f"{config.run.name_prefix}_{algorithm}_"
        f"f{config.model.force.num_bins}"
        f"-fmax{_number(config.binning.force_max_error)}"
        f"-fw{_number(config.loss.force_coefficient)}"
        f"-fmlp{_hidden_dims(config.model.force.hidden_dims)}_"
        f"e{config.model.energy.num_bins}"
        f"-emax{_number(config.binning.energy_max_error)}"
        f"-ew{_number(config.loss.energy_coefficient)}"
        f"-emlp{_hidden_dims(config.model.energy.hidden_dims)}"
        f"-order{config.model.energy.cumulant_order}"
    )
    if not _SAFE_RUN_NAME.fullmatch(name):
        raise ValueError("derived run name contains unsafe characters")
    return name


def resolve_run_dir(config: ConfidenceConfig) -> Path:
    """Resolve the named run directory and prove it remains below output_root."""
    runs_root = (config.run.output_root / "runs").resolve()
    run_dir = (runs_root / build_run_name(config)).resolve()
    if not run_dir.is_relative_to(runs_root):
        raise ValueError("derived run directory escapes output root")
    return run_dir
