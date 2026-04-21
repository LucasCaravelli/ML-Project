# TODO: Load configs/base.yaml into a typed configuration object that the rest of the package consumes.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    seed: int


@dataclass(frozen=True)
class PathsConfig:
    corpus: Path
    artifacts_dir: Path
    results_dir: Path


@dataclass(frozen=True)
class ChunkingConfig:
    sizes: list[int]
    overlaps: dict[int, int]


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str
    device: str
    batch_size: int
    normalize: bool


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int
    fusion_constant: int


@dataclass(frozen=True)
class GenerationConfig:
    backend: str
    model_name: str
    max_new_tokens: int
    temperature: float


@dataclass(frozen=True)
class QAConfig:
    target_count: int
    initial_batch_size: int
    type_distribution: dict[str, float]


@dataclass(frozen=True)
class RouterConfig:
    feature_sets: list[str]
    model_names: list[str]
    cv_folds: int


@dataclass(frozen=True)
class PromptsConfig:
    qa_generation: Path
    answer: Path


@dataclass(frozen=True)
class Config:
    project: ProjectConfig
    paths: PathsConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig
    qa: QAConfig
    router: RouterConfig
    prompts: PromptsConfig


def _read_yaml(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Config file {config_path} did not load as a dictionary.")

    return data


def load_config(config_path: str | Path = "configs/base.yaml") -> Config:
    config_path = Path(config_path)
    raw = _read_yaml(config_path)

    return Config(
        project=ProjectConfig(**raw["project"]),
        paths=PathsConfig(
            corpus=Path(raw["paths"]["corpus"]),
            artifacts_dir=Path(raw["paths"]["artifacts_dir"]),
            results_dir=Path(raw["paths"]["results_dir"]),
        ),
        chunking=ChunkingConfig(**raw["chunking"]),
        embedding=EmbeddingConfig(**raw["embedding"]),
        retrieval=RetrievalConfig(**raw["retrieval"]),
        generation=GenerationConfig(**raw["generation"]),
        qa=QAConfig(**raw["qa"]),
        router=RouterConfig(**raw["router"]),
        prompts=PromptsConfig(
            qa_generation=Path(raw["prompts"]["qa_generation"]),
            answer=Path(raw["prompts"]["answer"]),
        ),
    )