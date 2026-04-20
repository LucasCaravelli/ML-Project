from __future__ import annotations

from typing import TypedDict


class Chunk(TypedDict):
    chunk_id: str
    size: int
    start_char: int
    end_char: int
    text: str


class RetrievedChunk(TypedDict):
    chunk_id: str
    text: str
    score: float
    source_size: int


class QAPair(TypedDict):
    qa_id: str
    question: str
    answer: str
    source_chunk_id: str
    type: str
    validated: bool


class OracleLabel(TypedDict):
    qa_id: str
    best_size: int
    scores_by_size: dict[int, float]


class ScoreDict(TypedDict):
    em: float
    f1: float
    faithfulness: float
    cost_tokens: int
