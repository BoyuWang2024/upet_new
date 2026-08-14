from __future__ import annotations

from pathlib import Path

from test_migration import _config, _write_legacy_tree


def test_validate_migrated_run_checks_every_chunk_slice(tmp_path: Path) -> None:
    from Uncertainty_Quantification.BootStrapping.internal_migration.migration.converter import (
        convert_legacy_run,
    )
    from Uncertainty_Quantification.BootStrapping.internal_migration.migration.validation import (
        validate_migrated_run,
    )

    source = _write_legacy_tree(tmp_path / "source")
    destination = tmp_path / "destination"
    config = _config()
    convert_legacy_run(source, destination, config, tmp_path / "audit")

    result = validate_migrated_run(source, destination, config)

    assert result.member_count == 2
    assert result.validated_chunks == 8
    assert result.validated_checkpoints == 6
