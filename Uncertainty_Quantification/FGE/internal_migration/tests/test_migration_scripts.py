from __future__ import annotations

import pytest

from Uncertainty_Quantification.FGE.internal_migration.scripts import (
    audit_results,
    inspect_legacy,
    migrate_results,
    validate_migration,
)


@pytest.mark.parametrize(
    "module,arguments",
    [
        (inspect_legacy, []),
        (audit_results, []),
        (migrate_results, []),
        (validate_migration, []),
    ],
)
def test_migration_scripts_require_explicit_paths(module, arguments) -> None:
    with pytest.raises(SystemExit) as error:
        module.build_parser().parse_args(arguments)
    assert error.value.code == 2


def test_migrate_script_forwards_all_explicit_arguments(monkeypatch) -> None:
    seen = {}

    def fake_load(path):
        seen["config"] = path
        return "CONFIG"

    def fake_convert(source, destination, audit_root, config, base_checkpoint):
        seen.update(
            source=source,
            destination=destination,
            audit_root=audit_root,
            loaded=config,
            base_checkpoint=base_checkpoint,
        )

    monkeypatch.setattr(migrate_results, "load_config", fake_load)
    monkeypatch.setattr(migrate_results, "convert_legacy_run", fake_convert)
    assert (
        migrate_results.main(
            [
                "--source",
                "/old",
                "--destination",
                "/new",
                "--audit-root",
                "/audit",
                "--config",
                "/config.yaml",
                "--base-checkpoint",
                "/base.ckpt",
            ]
        )
        == 0
    )
    assert seen == {
        "config": "/config.yaml",
        "source": migrate_results.Path("/old"),
        "destination": migrate_results.Path("/new"),
        "audit_root": migrate_results.Path("/audit"),
        "loaded": "CONFIG",
        "base_checkpoint": migrate_results.Path("/base.ckpt"),
    }


def test_hard_failure_becomes_nonzero_exit(monkeypatch, capsys) -> None:
    from Uncertainty_Quantification.FGE.fge.errors import HardFailure

    def fail(*args, **kwargs):
        raise HardFailure("bad legacy source")

    monkeypatch.setattr(inspect_legacy, "inspect", fail)
    assert inspect_legacy.main(["--source", "/old"]) == 2
    assert capsys.readouterr().err.strip() == "bad legacy source"
