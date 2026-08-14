from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def test_atomic_writes_are_idempotent_but_never_clobber(tmp_path: Path) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.artifacts import (
        atomic_write_json,
        atomic_write_npz,
    )
    from Uncertainty_Quantification.BootStrapping.bootstrap.errors import HardFailure

    json_path = tmp_path / "artifact.json"
    atomic_write_json(json_path, {"value": 1})
    first = json_path.read_bytes()
    atomic_write_json(json_path, {"value": 1})
    assert json_path.read_bytes() == first
    assert json.loads(first) == {"value": 1}
    with pytest.raises(HardFailure, match="already exists"):
        atomic_write_json(json_path, {"value": 2})

    npz_path = tmp_path / "arrays.npz"
    atomic_write_npz(npz_path, values=np.arange(4, dtype=np.int64))
    atomic_write_npz(npz_path, values=np.arange(4, dtype=np.int64))
    with pytest.raises(HardFailure, match="already exists"):
        atomic_write_npz(npz_path, values=np.arange(5, dtype=np.int64))


def test_layout_is_stable_and_rejects_symlink_root(tmp_path: Path) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.artifacts import (
        ExperimentLayout,
    )
    from Uncertainty_Quantification.BootStrapping.bootstrap.errors import HardFailure

    layout = ExperimentLayout(tmp_path / "run")
    assert layout.member_dir(3) == (tmp_path / "run" / "members" / "member_003")
    assert layout.split_dir("val") == (tmp_path / "run" / "predictions" / "val")
    with pytest.raises(HardFailure, match="member_index"):
        layout.member_dir(-1)
    with pytest.raises(HardFailure, match="split"):
        layout.split_dir("../outside")

    outside = tmp_path / "outside"
    outside.mkdir()
    symlink = tmp_path / "linked-run"
    symlink.symlink_to(outside, target_is_directory=True)
    with pytest.raises(HardFailure, match="symlink"):
        ExperimentLayout(symlink)


def test_copy_file_exact_checks_existing_content(tmp_path: Path) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.artifacts import (
        copy_file_exact,
        sha256_file,
    )
    from Uncertainty_Quantification.BootStrapping.bootstrap.errors import HardFailure

    source = tmp_path / "source.bin"
    destination = tmp_path / "nested" / "destination.bin"
    source.write_bytes(b"checkpoint")
    copy_file_exact(source, destination)
    copy_file_exact(source, destination)
    assert sha256_file(source) == sha256_file(destination)

    source.write_bytes(b"different")
    with pytest.raises(HardFailure, match="already exists"):
        copy_file_exact(source, destination)
