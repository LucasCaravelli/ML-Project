from __future__ import annotations

from typing import Protocol

from .config import Config, load_config
from .generation import generate
from .retrieval import Retriever


class System(Protocol):
    name: str

    def answer(self, query: str, qa_id: str | None = None) -> tuple[str, list[str]]:
        """Return (prediction, passages_used) for a query."""
        ...


class _BaseSystem:
    def __init__(self, name: str, config: Config) -> None:
        self.name = name
        self._config = config
        self._retriever = Retriever(config)

    def _retrieve(self, query: str, chunk_size: int | str) -> list[str]:
        k = self._config.retrieval.top_k
        try:
            chunks = self._retriever.retrieve(query, chunk_size, k)  # type: ignore[arg-type]
            return [c["text"] for c in chunks]
        except (FileNotFoundError, RuntimeError):
            return [f"[MOCK] Passage {i + 1} for query: {query!r}" for i in range(k)]

    def _predict(self, query: str, passages: list[str]) -> str:
        try:
            return generate(query, passages, config=self._config)
        except ImportError:
            # Optional backend (e.g. ollama) not installed; caller should configure
            # an available backend such as 'extractive'.
            raise


class FixedSizeSystem(_BaseSystem):
    """Retrieve from a single chunk-size index."""

    def __init__(self, chunk_size: int, config: Config) -> None:
        super().__init__(f"fixed_{chunk_size}", config)
        self.chunk_size = chunk_size

    def answer(self, query: str, qa_id: str | None = None) -> tuple[str, list[str]]:
        passages = self._retrieve(query, self.chunk_size)
        return self._predict(query, passages), passages


class FusionSystem(_BaseSystem):
    """Retrieve via multi-scale reciprocal-rank fusion."""

    def __init__(self, config: Config) -> None:
        super().__init__("fusion", config)

    def answer(self, query: str, qa_id: str | None = None) -> tuple[str, list[str]]:
        passages = self._retrieve(query, "fusion")
        return self._predict(query, passages), passages


class OracleSystem(_BaseSystem):
    """Always retrieve from the best chunk size per query (from oracle labels).

    Requires oracle labels at artifacts/qa/oracle_labels.jsonl.
    Falls back to fixed_512 retrieval until labels are available.
    """

    def __init__(self, config: Config) -> None:
        super().__init__("oracle", config)
        self._labels: dict[str, int] = {}
        labels_path = config.paths.artifacts_dir / "qa" / "oracle_labels.jsonl"
        if labels_path.exists():
            from .io import read_jsonl
            for row in read_jsonl(labels_path):
                self._labels[row["qa_id"]] = row["best_size"]

    def answer(self, query: str, qa_id: str | None = None) -> tuple[str, list[str]]:
        size = self._labels.get(qa_id or "", self._config.chunking.sizes[-1])
        passages = self._retrieve(query, size)
        return self._predict(query, passages), passages


class RouterSystem(_BaseSystem):
    """Use a trained router to pick the chunk size per query.

    Requires a trained router model. Falls back to fixed_512 retrieval until trained.
    """

    def __init__(self, config: Config) -> None:
        super().__init__("router", config)

    def answer(self, query: str, qa_id: str | None = None) -> tuple[str, list[str]]:
        fallback_size = self._config.chunking.sizes[-1]
        passages = self._retrieve(query, fallback_size)
        return self._predict(query, passages), passages


_REGISTRY: dict[str, type[_BaseSystem]] = {
    "fusion": FusionSystem,
    "oracle": OracleSystem,
    "router": RouterSystem,
}


def build_system(name: str, config: Config | None = None) -> System:
    """Construct a named system (fixed_<size>, fusion, oracle, router)."""
    if config is None:
        config = load_config()

    if name.startswith("fixed_"):
        size = int(name.split("_", 1)[1])
        return FixedSizeSystem(size, config)

    cls = _REGISTRY.get(name)
    if cls is None:
        valid = list(_REGISTRY) + ["fixed_<size>"]
        raise ValueError(f"Unknown system {name!r}. Valid names: {valid}")
    return cls(config)
