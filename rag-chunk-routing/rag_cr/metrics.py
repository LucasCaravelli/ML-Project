from __future__ import annotations

import re
import string
from collections import Counter

from .tokens import count_tokens
from .types import ScoreDict


def _normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(text: str) -> list[str]:
    return _normalize(text).split()


def _exact_match(prediction: str, gold: str) -> float:
    return 1.0 if _normalize(prediction) == _normalize(gold) else 0.0


def token_f1(prediction: str, gold: str) -> float:
    pred_counts = Counter(_tokenize(prediction))
    gold_counts = Counter(_tokenize(gold))

    common = sum((pred_counts & gold_counts).values())
    if common == 0:
        return 0.0

    precision = common / sum(pred_counts.values())
    recall = common / sum(gold_counts.values())
    return 2 * precision * recall / (precision + recall)


def _faithfulness(prediction: str, passages: list[str]) -> float:
    """Fraction of unique prediction tokens that appear in the retrieved passages."""
    pred_tokens = set(_tokenize(prediction))
    if not pred_tokens or not passages:
        return 0.0
    passage_tokens = {tok for p in passages for tok in _tokenize(p)}
    return len(pred_tokens & passage_tokens) / len(pred_tokens)


def score(prediction: str, gold: str, passages: list[str]) -> ScoreDict:
    """Compute deterministic evaluation metrics for one prediction."""
    return ScoreDict(
        em=_exact_match(prediction, gold),
        f1=token_f1(prediction, gold),
        faithfulness=_faithfulness(prediction, passages),
        cost_tokens=sum(count_tokens(p) for p in passages),
    )
