from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def tmp_output(tmp_path: Path) -> Path:
    output = tmp_path / "outputs"
    output.mkdir()
    return output
