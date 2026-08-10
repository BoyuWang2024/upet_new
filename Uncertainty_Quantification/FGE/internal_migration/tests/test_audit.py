from __future__ import annotations

import json
from pathlib import Path

import pytest

from Uncertainty_Quantification.FGE.fge.errors import HardFailure
from Uncertainty_Quantification.FGE.internal_migration.migration import converter

from .helpers import build_conversion_case


def test_audit_is_durable_before_final_rename(
    legacy_tree, config_payload, tmp_path: Path, monkeypatch
) -> None:
    source, base, config, expected = build_conversion_case(
        legacy_tree, config_payload, tmp_path
    )
    destination = tmp_path / "published"
    audit_root = tmp_path / "audit"
    real_replace = converter.os.replace

    def fail_final(source_path, destination_path, **kwargs):
        bound_parent = (Path("/proc/self/fd") / str(kwargs.get("dst_dir_fd"))).resolve()
        if bound_parent == destination.parent:
            assert (audit_root / "published/audit.json").is_file()
            raise OSError("injected final rename failure")
        return real_replace(source_path, destination_path, **kwargs)

    monkeypatch.setattr(converter.os, "replace", fail_final)
    with pytest.raises(HardFailure, match="publish"):
        converter.convert_legacy_run(
            source, destination, audit_root, config, base, expected=expected
        )
    assert not destination.exists()
    audit = json.loads((audit_root / "published/audit.json").read_text())
    assert audit["publication_authorized"] is True


def test_audit_failure_prevents_formal_publication(
    legacy_tree, config_payload, tmp_path: Path, monkeypatch
) -> None:
    source, base, config, expected = build_conversion_case(
        legacy_tree, config_payload, tmp_path
    )
    destination = tmp_path / "published"

    def fail(*args, **kwargs):
        raise HardFailure("injected audit failure")

    monkeypatch.setattr(converter, "write_external_audit", fail)
    with pytest.raises(HardFailure, match="audit"):
        converter.convert_legacy_run(
            source,
            destination,
            tmp_path / "audit",
            config,
            base,
            expected=expected,
        )
    assert not destination.exists()
