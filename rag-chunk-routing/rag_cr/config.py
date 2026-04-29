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
class QAModelsConfig:
    factoid_question: str
    multihop_question: str
    synthesis_question: str
    answer: str
    judge: str


@dataclass(frozen=True)
class QAGenerationConfig:
    provider: str
    models: QAModelsConfig
    question_temperature: float
    answer_temperature: float
    judge_temperature: float
    max_tokens: int
    request_timeout_s: int
    max_retries: int
    primary_only_f1_threshold: float
    max_topup_rounds: int


@dataclass(frozen=True)
class QAConfig:
    target_count: int
    initial_batch_size: int
    source_chunk_size: int
    neighbor_window: int
    type_distribution: dict[str, float]
    generation: QAGenerationConfig


@dataclass(frozen=True)
class RouterConfig:
    feature_sets: list[str]
    model_names: list[str]
    cv_folds: int


@dataclass(frozen=True)
class PromptsConfig:
    qa_generation_factoid: Path
    qa_generation_multihop: Path
    qa_generation_synthesis: Path
    qa_answer: Path
    answer: Path
    judge: Path


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
        qa=QAConfig(
            target_count=raw["qa"]["target_count"],
            initial_batch_size=raw["qa"]["initial_batch_size"],
            source_chunk_size=raw["qa"]["source_chunk_size"],
            neighbor_window=raw["qa"]["neighbor_window"],
            type_distribution=raw["qa"]["type_distribution"],
            generation=QAGenerationConfig(
                provider=raw["qa"]["generation"]["provider"],
                models=QAModelsConfig(**raw["qa"]["generation"]["models"]),
                question_temperature=raw["qa"]["generation"]["question_temperature"],
                answer_temperature=raw["qa"]["generation"]["answer_temperature"],
                judge_temperature=raw["qa"]["generation"]["judge_temperature"],
                max_tokens=raw["qa"]["generation"]["max_tokens"],
                request_timeout_s=raw["qa"]["generation"]["request_timeout_s"],
                max_retries=raw["qa"]["generation"]["max_retries"],
                primary_only_f1_threshold=raw["qa"]["generation"]["primary_only_f1_threshold"],
                max_topup_rounds=raw["qa"]["generation"]["max_topup_rounds"],
            ),
        ),
        router=RouterConfig(**raw["router"]),
        prompts=PromptsConfig(
            qa_generation_factoid=Path(raw["prompts"]["qa_generation_factoid"]),
            qa_generation_multihop=Path(raw["prompts"]["qa_generation_multihop"]),
            qa_generation_synthesis=Path(raw["prompts"]["qa_generation_synthesis"]),
            qa_answer=Path(raw["prompts"]["qa_answer"]),
            answer=Path(raw["prompts"]["answer"]),
            judge=Path(raw["prompts"]["judge"]),
        ),
    )