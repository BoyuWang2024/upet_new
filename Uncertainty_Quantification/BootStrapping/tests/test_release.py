from __future__ import annotations

import tarfile
from pathlib import Path


def test_release_archive_includes_formal_code_and_excludes_adapter(
    tmp_path: Path,
) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.release import (
        build_source_release,
    )

    source = tmp_path / "BootStrapping"
    for relative in (
        "bootstrap/core.py",
        "scripts/run.py",
        "configs/run.yaml",
        "internal_migration/secret.py",
        "tests/test_core.py",
        "outputs/result.npz",
    ):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    (source / "README.md").write_text("public", encoding="utf-8")
    (source / "__init__.py").write_text("", encoding="utf-8")

    archive = build_source_release(source, tmp_path / "bootstrap-release.tar.gz")

    with tarfile.open(archive, "r:gz") as handle:
        names = set(handle.getnames())
    assert "BootStrapping/bootstrap/core.py" in names
    assert "BootStrapping/scripts/run.py" in names
    assert "BootStrapping/configs/run.yaml" in names
    assert "BootStrapping/README.md" in names
    assert all("internal_migration" not in name for name in names)
    assert all("/tests/" not in name for name in names)
    assert all("/outputs/" not in name for name in names)
