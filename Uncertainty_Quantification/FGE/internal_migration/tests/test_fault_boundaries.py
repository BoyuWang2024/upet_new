from __future__ import annotations

import os
from pathlib import Path

import pytest

from Uncertainty_Quantification.FGE.fge import artifacts, validation
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


def test_failed_converter_publication_retains_bound_staging_after_ancestor_swap(
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
    real_rename = artifacts._rename_directory_noreplace
    observed: dict[str, str] = {}

    def swap_and_fail(parent_fd: int, source_name: str, target_name: str) -> None:
        bound_parent = (Path("/proc/self/fd") / str(parent_fd)).resolve()
        if bound_parent != publication:
            return real_rename(parent_fd, source_name, target_name)
        os.rename(publication, owned)
        publication.symlink_to(outside, target_is_directory=True)
        observed["staging"] = source_name
        raise OSError("injected publication failure")

    monkeypatch.setattr(artifacts, "_rename_directory_noreplace", swap_and_fail)
    with pytest.raises(HardFailure, match="descriptor-bound staging"):
        converter.convert_legacy_run(
            source,
            destination,
            tmp_path / "audit_dynamic",
            config,
            base,
            expected=expected,
        )

    assert (owned / observed["staging"]).is_dir()
    assert not destination.exists()
    assert not any(outside.iterdir())


def test_converter_publication_rejects_configured_parent_swap(
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
    real_rename = artifacts._rename_directory_noreplace

    def swap_then_rename(parent_fd: int, source_name: str, target_name: str) -> None:
        bound_parent = (Path("/proc/self/fd") / str(parent_fd)).resolve()
        if bound_parent != publication:
            return real_rename(parent_fd, source_name, target_name)
        os.rename(publication, owned)
        publication.symlink_to(outside, target_is_directory=True)
        real_rename(parent_fd, source_name, target_name)

    monkeypatch.setattr(artifacts, "_rename_directory_noreplace", swap_then_rename)
    with pytest.raises(HardFailure, match="descriptor-bound staging"):
        converter.convert_legacy_run(
            source,
            destination,
            tmp_path / "audit_bound",
            config,
            base,
            expected=expected,
        )

    assert not (outside / destination.name).exists()
    assert (owned / destination.name / "result_manifest.json").is_file()


def test_converter_rejects_staging_basename_substitution_without_acceptance(
    legacy_tree, config_payload, tmp_path: Path, monkeypatch
) -> None:
    source, base, config, expected = build_conversion_case(
        legacy_tree, config_payload, tmp_path
    )
    destination = tmp_path / "published_basename"
    real_rename = artifacts._rename_directory_noreplace
    observed: dict[str, str] = {}

    def substitute(parent_fd: int, source_name: str, target_name: str) -> None:
        bound_parent = (Path("/proc/self/fd") / str(parent_fd)).resolve()
        if bound_parent != tmp_path:
            return real_rename(parent_fd, source_name, target_name)
        owned = f".owned-{source_name}"
        os.rename(source_name, owned, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        os.mkdir(source_name, 0o700, dir_fd=parent_fd)
        attacker_fd = os.open(source_name, artifacts._DIRECTORY_FLAGS, dir_fd=parent_fd)
        try:
            marker = os.open(
                "attacker.txt",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=attacker_fd,
            )
            os.write(marker, b"attacker")
            os.close(marker)
        finally:
            os.close(attacker_fd)
        observed["owned"] = owned
        real_rename(parent_fd, source_name, target_name)

    monkeypatch.setattr(artifacts, "_rename_directory_noreplace", substitute)
    with pytest.raises(HardFailure, match="descriptor-bound staging|identity differs"):
        converter.convert_legacy_run(
            source,
            destination,
            tmp_path / "audit_basename",
            config,
            base,
            expected=expected,
        )

    assert (destination / "attacker.txt").read_bytes() == b"attacker"
    assert not (destination / "result_manifest.json").exists()
    assert (tmp_path / observed["owned"] / "result_manifest.json").is_file()
