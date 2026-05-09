from __future__ import annotations

from pathlib import Path

import pytest

from rag_cr.config import load_config
from rag_cr.retrieval import Retriever

# Every test in this file requires a built artifacts/ directory (FAISS indices +
# chunk files produced by `make indices`). Run `pytest -m "not integration"` to
# skip them on a fresh checkout.
pytestmark = pytest.mark.integration

QUERY = "What is machine learning?"


@pytest.fixture
def retriever(project_root: Path) -> Retriever:
    indices_dir = project_root / "artifacts" / "indices"
    if not indices_dir.exists() or not any(indices_dir.glob("*.faiss")):
        pytest.skip("Built artifacts/indices/ required; run `make indices` first.")
    cfg = load_config(project_root / "configs" / "base.yaml")
    return Retriever(cfg, artifacts_dir=project_root / "artifacts")


# ---------------------------------------------------------------------------
# Single-scale retrieval
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [128, 256, 512, 1024])
def test_single_scale_returns_k_chunks(retriever: Retriever, size: int) -> None:
    results = retriever.retrieve(QUERY, chunk_size=size, k=5)
    assert len(results) == 5


@pytest.mark.parametrize("size", [128, 256, 512, 1024])
def test_single_scale_scores_nonincreasing(retriever: Retriever, size: int) -> None:
    results = retriever.retrieve(QUERY, chunk_size=size, k=5)
    scores = [c["score"] for c in results]
    assert scores == sorted(scores, reverse=True), "Scores must be in non-increasing order"


@pytest.mark.parametrize("size", [128, 256, 512, 1024])
def test_single_scale_chunk_ids_match_size(retriever: Retriever, size: int) -> None:
    results = retriever.retrieve(QUERY, chunk_size=size, k=5)
    for chunk in results:
        assert chunk["chunk_id"].startswith(f"{size}-"), (
            f"chunk_id {chunk['chunk_id']!r} does not start with '{size}-'"
        )


@pytest.mark.parametrize("size", [128, 256, 512, 1024])
def test_single_scale_result_fields(retriever: Retriever, size: int) -> None:
    results = retriever.retrieve(QUERY, chunk_size=size, k=3)
    for chunk in results:
        assert isinstance(chunk["chunk_id"], str) and chunk["chunk_id"]
        assert isinstance(chunk["text"], str) and chunk["text"]
        assert isinstance(chunk["score"], float)
        assert chunk["score"] > 0
        assert chunk["source_size"] == size


def test_single_scale_respects_k(retriever: Retriever) -> None:
    for k in (1, 3, 5):
        results = retriever.retrieve(QUERY, chunk_size=256, k=k)
        assert len(results) == k


def test_single_scale_no_duplicate_chunk_ids(retriever: Retriever) -> None:
    results = retriever.retrieve(QUERY, chunk_size=256, k=5)
    ids = [c["chunk_id"] for c in results]
    assert len(ids) == len(set(ids)), "Returned chunk IDs must be unique"


# ---------------------------------------------------------------------------
# Fusion retrieval
# ---------------------------------------------------------------------------


def test_fusion_returns_k_chunks(retriever: Retriever) -> None:
    results = retriever.retrieve(QUERY, chunk_size="fusion", k=5)
    assert len(results) == 5


def test_fusion_scores_positive(retriever: Retriever) -> None:
    results = retriever.retrieve(QUERY, chunk_size="fusion", k=5)
    for chunk in results:
        assert chunk["score"] > 0


def test_fusion_scores_nonincreasing(retriever: Retriever) -> None:
    results = retriever.retrieve(QUERY, chunk_size="fusion", k=5)
    scores = [c["score"] for c in results]
    assert scores == sorted(scores, reverse=True), "Fusion scores must be in non-increasing order"


def test_fusion_covers_multiple_sizes(retriever: Retriever) -> None:
    results = retriever.retrieve(QUERY, chunk_size="fusion", k=12)
    sizes_present = {c["source_size"] for c in results}
    assert len(sizes_present) > 1, "Fusion should draw from more than one chunk size"


def test_fusion_result_fields(retriever: Retriever) -> None:
    cfg_sizes = {128, 256, 512, 1024}
    results = retriever.retrieve(QUERY, chunk_size="fusion", k=5)
    for chunk in results:
        assert isinstance(chunk["chunk_id"], str) and chunk["chunk_id"]
        assert isinstance(chunk["text"], str) and chunk["text"]
        assert isinstance(chunk["score"], float)
        assert chunk["source_size"] in cfg_sizes


def test_fusion_no_duplicate_chunk_ids(retriever: Retriever) -> None:
    results = retriever.retrieve(QUERY, chunk_size="fusion", k=8)
    ids = [c["chunk_id"] for c in results]
    assert len(ids) == len(set(ids)), "Fusion must not return duplicate chunk IDs"


def test_fusion_respects_k(retriever: Retriever) -> None:
    for k in (1, 4, 8):
        results = retriever.retrieve(QUERY, chunk_size="fusion", k=k)
        assert len(results) == k


# ---------------------------------------------------------------------------
# Cross-size consistency
# ---------------------------------------------------------------------------


def test_different_queries_give_different_results(retriever: Retriever) -> None:
    r1 = retriever.retrieve("What is machine learning?", chunk_size=256, k=3)
    r2 = retriever.retrieve("How does monetary policy work?", chunk_size=256, k=3)
    ids1 = {c["chunk_id"] for c in r1}
    ids2 = {c["chunk_id"] for c in r2}
    assert ids1 != ids2, "Different queries should retrieve different chunks"
