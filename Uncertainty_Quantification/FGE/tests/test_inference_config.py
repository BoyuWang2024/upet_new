from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from Uncertainty_Quantification.FGE.fge.errors import HardFailure
from Uncertainty_Quantification.FGE.fge.inference_config import load_inference_config


MANIFEST_SHA = "b1f4a3c5c7713b4be369b467e98e62d59bc23a13aefceffbd49ac51773b6dc86"
CHECKPOINT_SHA = "879b1045391d88869522605a8b8b3cedeed74668e7062fdd7487548ab7b08004"
MAD_SHA = "d9a1280246a7a678f699e7654aebd29e4273ab6dcd1dfb4f74334a9b15edb66b"


def _payload() -> dict[str, object]:
    return {
        "schema_version": "upet.fge.inference.v1",
        "ensemble": {
            "root": "ensemble",
            "result_manifest_sha256": MANIFEST_SHA,
            "base_checkpoint": "checkpoints/base.ckpt",
            "base_checkpoint_sha256": CHECKPOINT_SHA,
            "member_count": 8,
        },
        "dataset": {
            "label": "mad_test",
            "path": "data/mad-test.xyz",
            "expected_sha256": MAD_SHA,
            "split": "test",
            "reference_availability": {
                "energy": True,
                "forces": True,
                "stress": False,
            },
        },
        "chunking": {"max_structures": 128, "max_atoms": 4096},
        "output": {"root": "outputs/mad_test"},
        "runtime": {"device": "cpu", "torch_threads": 16},
    }


def _write(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "configs" / "inference.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_inference_config_loads_exact_contract(tmp_path: Path) -> None:
    path = _write(tmp_path, _payload())

    config = load_inference_config(path)

    assert config.dataset.label == "mad_test"
    assert config.dataset.expected_sha256 == MAD_SHA
    assert dict(config.dataset.reference_availability) == {
        "energy": True,
        "forces": True,
        "stress": False,
    }
    assert config.chunking.max_structures == 128
    assert config.chunking.max_atoms == 4096
    assert config.ensemble.member_count == 8
    assert config.runtime.device == "cpu"


def test_inference_config_resolves_relative_runtime_paths(tmp_path: Path) -> None:
    path = _write(tmp_path, _payload())

    config = load_inference_config(path)

    base = path.parent
    assert config.ensemble.root == (base / "ensemble").resolve()
    assert config.ensemble.base_checkpoint == (base / "checkpoints/base.ckpt").resolve()
    assert config.dataset.path == (base / "data/mad-test.xyz").resolve()
    assert config.output.root == (base / "outputs/mad_test").resolve()


def test_sanitized_config_is_path_neutral(tmp_path: Path) -> None:
    first_path = _write(tmp_path / "first", _payload())
    second_path = _write(tmp_path / "second", _payload())

    first = load_inference_config(first_path).sanitized()
    second = load_inference_config(second_path).sanitized()

    assert first == second
    rendered = repr(first)
    assert str(tmp_path) not in rendered
    assert "base.ckpt" not in rendered
    assert "mad-test.xyz" not in rendered


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("ensemble", "member_count", 7, "ensemble"),
        ("ensemble", "result_manifest_sha256", "A" * 64, "ensemble"),
        ("dataset", "label", "MAD-test", "dataset"),
        ("chunking", "max_structures", 0, "chunking"),
        ("chunking", "max_atoms", True, "chunking"),
        ("runtime", "device", "cuda", "runtime"),
        ("runtime", "torch_threads", 0, "runtime"),
    ],
)
def test_inference_config_rejects_invalid_contract_values(
    tmp_path: Path,
    section: str,
    key: str,
    value: object,
    message: str,
) -> None:
    payload = deepcopy(_payload())
    target = payload[section]
    assert isinstance(target, dict)
    target[key] = value

    with pytest.raises(HardFailure, match=message):
        load_inference_config(_write(tmp_path, payload))


@pytest.mark.parametrize("section", [None, "ensemble", "dataset", "runtime"])
def test_inference_config_rejects_unknown_keys(
    tmp_path: Path, section: str | None
) -> None:
    payload = deepcopy(_payload())
    target = payload if section is None else payload[section]
    assert isinstance(target, dict)
    target["unexpected"] = "forbidden"

    with pytest.raises(HardFailure, match="unknown key"):
        load_inference_config(_write(tmp_path, payload))


@pytest.mark.parametrize(
    ("availability", "message"),
    [
        ({"energy": True, "forces": True}, "reference_availability"),
        (
            {"energy": True, "forces": True, "stress": False, "virial": False},
            "reference_availability",
        ),
        (
            {"energy": True, "forces": 1, "stress": False},
            "reference_availability",
        ),
    ],
)
def test_inference_config_requires_exact_strict_boolean_references(
    tmp_path: Path, availability: dict[str, object], message: str
) -> None:
    payload = deepcopy(_payload())
    dataset = payload["dataset"]
    assert isinstance(dataset, dict)
    dataset["reference_availability"] = availability

    with pytest.raises(HardFailure, match=message):
        load_inference_config(_write(tmp_path, payload))
