from pathlib import Path

import pytest

from Uncertainty_Quantification.FGE.fge.errors import HardFailure
from Uncertainty_Quantification.FGE.internal_migration.migration.converter import (
    convert_legacy_run,
)

from .helpers import build_conversion_case


def test_converter_rejects_broken_destination_symlink(
    legacy_tree, config_payload, tmp_path: Path
) -> None:
    source, base, config, expected = build_conversion_case(
        legacy_tree, config_payload, tmp_path
    )
    destination = tmp_path / "published"
    destination.symlink_to(tmp_path / "missing-target", target_is_directory=True)

    with pytest.raises(HardFailure, match="destination"):
        convert_legacy_run(
            source,
            destination,
            tmp_path / "audit",
            config,
            base,
            expected=expected,
        )
