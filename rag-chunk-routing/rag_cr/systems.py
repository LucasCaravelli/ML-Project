# TODO: Implement one system class per evaluated configuration (fixed-size, fusion, oracle, router), each exposing the same answer-a-query interface so pipeline.py can treat them uniformly.

from __future__ import annotations

from typing import Protocol


class System(Protocol):
    name: str

    def answer(self, query: str) -> tuple[str, list[str]]:
        """Return (prediction, passages_used) for a query."""
        ...


def build_system(name: str) -> System:
    """Construct a named system (fixed-size, fusion, oracle, router)."""
    raise NotImplementedError
