from pathlib import Path

from confidence_head.external_config import load_external_config


CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_production_separates_upet_extraction_and_head_batch_sizes() -> None:
    config = load_external_config(CONFIGS / "predict_external_gpu.yaml")

    assert config.cache_batch_size == 2
    assert config.batch_size == 128


def test_smoke_uses_small_cpu_extraction_batches() -> None:
    config = load_external_config(CONFIGS / "predict_external_smoke.yaml")

    assert config.cache_batch_size == 2
    assert config.batch_size == 4
