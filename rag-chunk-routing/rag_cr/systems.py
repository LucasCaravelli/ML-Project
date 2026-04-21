from __future__ import annotations

from typing import Protocol

from .config import Config, load_config


class System(Protocol):
    name: str

    def answer(self, query: str) -> tuple[str, list[str]]:
        """Return (prediction, passages_used) for a query."""
        ...


class _BaseSystem:
    """Shared mock retrieval and generation used until real indices are built."""

    def __init__(self, name: str, config: Config) -> None:
        self.name = name
        self._config = config

    def _mock_passages(self, query: str) -> list[str]:
        k = self._config.retrieval.top_k
        return [f"[MOCK] Passage {i + 1} for query: {query!r}" for i in range(k)]

    def _mock_predict(self, query: str) -> str:
        return f"[MOCK] Answer to: {query!r}"


class FixedSizeSystem(_BaseSystem):
    """Retrieve from a single chunk-size index."""

    def __init__(self, chunk_size: int, config: Config) -> None:
        super().__init__(f"fixed_{chunk_size}", config)
        self.chunk_size = chunk_size

    def answer(self, query: str) -> tuple[str, list[str]]:
        passages = self._mock_passages(query)
        return self._mock_predict(query), passages


class FusionSystem(_BaseSystem):
    """Retrieve via multi-scale reciprocal-rank fusion."""

    def __init__(self, config: Config) -> None:
        super().__init__("fusion", config)

    def answer(self, query: str) -> tuple[str, list[str]]:
        passages = self._mock_passages(query)
        return self._mock_predict(query), passages


class OracleSystem(_BaseSystem):
    """Always retrieve from the best chunk size per query (from oracle labels)."""

    def __init__(self, config: Config) -> None:
        super().__init__("oracle", config)

    def answer(self, query: str) -> tuple[str, list[str]]:
        passages = self._mock_passages(query)
        return self._mock_predict(query), passages


class RouterSystem(_BaseSystem):
    """Use a trained router to pick the chunk size per query."""

    def __init__(self, config: Config) -> None:
        super().__init__("router", config)

    def answer(self, query: str) -> tuple[str, list[str]]:
        passages = self._mock_passages(query)
        return self._mock_predict(query), passages


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
