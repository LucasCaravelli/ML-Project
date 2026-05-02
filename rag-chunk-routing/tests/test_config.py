from __future__ import annotations

from pathlib import Path

from rag_cr.config import Config, load_config


def test_load_base_config(project_root: Path) -> None:
    config = load_config(project_root / "configs" / "base.yaml")

    assert isinstance(config, Config)
    assert config.project.seed == 42
    assert config.chunking.sizes == [128, 256, 512]
    assert 0.99 < sum(config.qa.type_distribution.values()) < 1.01
