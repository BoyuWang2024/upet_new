from __future__ import annotations

from pathlib import Path

import pytest

from Uncertainty_Quantification.FGE.internal_migration.migration import converter

from .helpers import build_conversion_case


def test_converter_rejects_even_a_private_restart_materializer_override(
    legacy_tree, config_payload, tmp_path: Path
) -> None:
    source, base, config, expected = build_conversion_case(
        legacy_tree, config_payload, tmp_path
    )

    with pytest.raises(TypeError, match="_restart_materializer"):
        converter.convert_legacy_run(
            source,
            tmp_path / "published",
            tmp_path / "audit",
            config,
            base,
            expected=expected,
            _restart_materializer=lambda path, digest: {},  # type: ignore[call-arg]
        )
