from __future__ import annotations

import os
from pathlib import Path

import pytest

from Uncertainty_Quantification.FGE.fge import validation
from Uncertainty_Quantification.FGE.fge.errors import HardFailure
from Uncertainty_Quantification.FGE.internal_migration.migration import converter

from .helpers import build_conversion_case


@pytest.mark.parametrize(
    "boundary", ["member", "prediction", "evaluation", "validation", "result_manifest"]
)
def test_every_formal_write_boundary_retains_staging_without_final(
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
        )

    assert not destination.exists()
    retained = tuple(tmp_path.glob(".published.staging-*"))
    assert len(retained) == 1
    assert retained[0].is_dir()


def test_failed_converter_publication_never_deletes_through_swapped_ancestor(
    legacy_tree, config_payload, tmp_path: Path, monkeypatch
) -> None:
    source, base, config, expected = build_conversion_case(
        legacy_tree, config_payload, tmp_path
    )
    publication = tmp_path / "publication"
    publication.mkdir()
    destination = publication / "published"
    owned = tmp_path / ".owned_publication"
    outside = tmp_path / "outside_publication"
    outside.mkdir()
    real_replace = converter.os.replace
    observed: dict[str, Path] = {}

    def swap_and_fail(source_path, destination_path, **kwargs):
        bound_parent = Path("/proc/self/fd") / str(kwargs.get("dst_dir_fd"))
        if bound_parent.resolve() != publication:
            return real_replace(source_path, destination_path, **kwargs)
        staging = Path(source_path)
        os.rename(publication, owned)
        publication.symlink_to(outside, target_is_directory=True)
        external = outside / staging.name
        external.mkdir()
        sentinel = external / "sentinel.txt"
        sentinel.write_text("outside", encoding="utf-8")
        observed["staging"] = staging
        observed["sentinel"] = sentinel
        raise OSError("injected publication failure")

    monkeypatch.setattr(converter.os, "replace", swap_and_fail)

    with pytest.raises(HardFailure, match="unable to publish descriptor-bound staging"):
        converter.convert_legacy_run(
            source,
            destination,
            tmp_path / "audit_dynamic",
            config,
            base,
            expected=expected,
        )

    staging = observed["staging"]
    sentinel = observed["sentinel"]
    assert sentinel.read_text(encoding="utf-8") == "outside"
    assert (owned / staging.name).is_dir()
    assert not destination.exists()


def test_converter_publication_remains_bound_when_parent_is_swapped(
    legacy_tree, config_payload, tmp_path: Path, monkeypatch
) -> None:
    source, base, config, expected = build_conversion_case(
        legacy_tree, config_payload, tmp_path
    )
    publication = tmp_path / "bound_publication"
    publication.mkdir()
    destination = publication / "published"
    owned = tmp_path / ".owned_bound_publication"
    outside = tmp_path / "outside_bound_publication"
    outside.mkdir()
    real_replace = converter.os.replace
    observed: dict[str, Path] = {}

    def swap_then_replace(source_path, destination_path, **kwargs):
        bound = kwargs.get("dst_dir_fd")
        if bound is None:
            targets_publication = Path(destination_path) == destination
        else:
            bound_parent = (Path("/proc/self/fd") / str(bound)).resolve()
            targets_publication = bound_parent == publication
        if not targets_publication:
            return real_replace(source_path, destination_path, **kwargs)
        os.rename(publication, owned)
        publication.symlink_to(outside, target_is_directory=True)
        attacker = outside / Path(source_path).name
        attacker.mkdir()
        (attacker / "attacker.txt").write_text("outside", encoding="utf-8")
        observed["attacker"] = attacker
        return real_replace(source_path, destination_path, **kwargs)

    monkeypatch.setattr(converter.os, "replace", swap_then_replace)
    result = converter.convert_legacy_run(
        source,
        destination,
        tmp_path / "audit_bound",
        config,
        base,
        expected=expected,
    )

    assert result == destination
    assert (observed["attacker"] / "attacker.txt").is_file()
    assert not (outside / destination.name).exists()
    assert (owned / destination.name / "result_manifest.json").is_file()
