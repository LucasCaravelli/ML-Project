# TODO: Implement deterministic evaluation metrics (exact match, token-level F1, passage-overlap faithfulness, retrieval cost in tokens). No LLM-as-judge scoring.

from __future__ import annotations

from .types import ScoreDict


def score(prediction: str, gold: str, passages: list[str]) -> ScoreDict:
    """Compute deterministic evaluation metrics for one prediction."""
    raise NotImplementedError
