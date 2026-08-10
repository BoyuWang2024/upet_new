from Uncertainty_Quantification.FGE import internal_migration


def test_internal_package_exports_migration_entry_points() -> None:
    assert callable(internal_migration.convert_legacy_run)
    assert callable(internal_migration.write_external_audit)
