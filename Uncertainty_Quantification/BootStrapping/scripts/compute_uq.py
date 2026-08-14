"""Compute canonical BootStrapping uncertainty from published predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..bootstrap.config import BootstrapConfig
from ..bootstrap.uncertainty import compute_store_uncertainty
from ._cli import run_stage


def _run(config: BootstrapConfig) -> None:
    run_root = (
        config.experiment.output_root
        / config.experiment.name
        / config.experiment.run_id
    )
    units = {
        "energy": config.data.units.energy,
        "forces": config.data.units.forces,
        "stress": config.data.units.stress,
    }
    for split in config.prediction.splits:
        for mode in config.uncertainty.parameter_modes:
            compute_store_uncertainty(
                run_root / "predictions",
                run_root / "uncertainty",
                split=split,
                mode=mode,
                member_count=config.bootstrap.ensemble_size,
                units=units,
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    return run_stage(parser, argv, _run)


if __name__ == "__main__":
    raise SystemExit(main())
