from __future__ import annotations

from pathlib import Path

import pytest

from Uncertainty_Quantification.FGE.fge import validation
from Uncertainty_Quantification.FGE.fge.errors import HardFailure
from Uncertainty_Quantification.FGE.internal_migration.migration import converter

from .helpers import build_conversion_case


@pytest.mark.parametrize(
    "boundary", ["member", "prediction", "evaluation", "validation", "result_manifest"]
)
def test_every_formal_write_boundary_cleans_staging_and_final(
    legacy_tree, config_payload, tmp_path: Path, monkeypatch, boundary: str
) -> None:
    source, base, config, expected = build_conversion_case(
        legacy_tree, config_payload, tmp_path
    )
    destination = tmp_path / "published"

    if boundary == "validation":
        monkeypatch.setattr(
            converter,
            "validate_result",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                HardFailure("injected validation write failure")
            ),
        )
    elif boundary == "result_manifest":
        real_json = validation.atomic_write_json

        def fail_manifest(path, value):
            if Path(path).name == "result_manifest.json":
                raise HardFailure("injected result_manifest write failure")
            return real_json(path, value)

        monkeypatch.setattr(validation, "atomic_write_json", fail_manifest)
    else:
        real_save = converter.atomic_torch_save

        def fail_selected(path, value):
            relative = Path(path).as_posix()
            selected = {
                "member": "/training/members/",
                "prediction": "/prediction/test_raw.pt",
                "evaluation": "/evaluation/legacy_equal_weight/ensemble.pt",
            }[boundary]
            if selected in relative:
                raise HardFailure(f"injected {boundary} write failure")
            return real_save(path, value)

        monkeypatch.setattr(converter, "atomic_torch_save", fail_selected)

    with pytest.raises(HardFailure, match="injected"):
        converter.convert_legacy_run(
            source,
            destination,
            tmp_path / "audit",
            config,
            base,
            expected=expected,
            code_identity={"commit": "a" * 40, "dirty_sha256": "b" * 64},
        )

    assert not destination.exists()
    assert not tuple(tmp_path.glob(".published.staging-*"))
