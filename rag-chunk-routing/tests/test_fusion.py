from __future__ import annotations

import pytest

from rag_cr.fusion import rrf_fuse
from rag_cr.types import RetrievedChunk


def _chunk(chunk_id: str, text: str = "x", source_size: int = 128) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, text=text, score=0.0, source_size=source_size)


def test_empty_input_returns_empty():
    assert rrf_fuse([], fusion_constant=60, top_k=5) == []


def test_empty_inner_lists_returns_empty():
    assert rrf_fuse([[], []], fusion_constant=60, top_k=5) == []


def test_single_ranking_preserves_order():
    ranked = [_chunk("a"), _chunk("b"), _chunk("c")]
    out = rrf_fuse([ranked], fusion_constant=60, top_k=3)
    assert [c["chunk_id"] for c in out] == ["a", "b", "c"]


def test_single_ranking_scores_use_fusion_constant():
    ranked = [_chunk("a"), _chunk("b")]
    out = rrf_fuse([ranked], fusion_constant=60, top_k=2)
    # rank starts at 1, so score = 1/(60+rank)
    assert out[0]["score"] == pytest.approx(1.0 / 61)
    assert out[1]["score"] == pytest.approx(1.0 / 62)


def test_overlapping_chunks_have_summed_scores():
    list_a = [_chunk("x"), _chunk("y")]
    list_b = [_chunk("y"), _chunk("x")]
    out = rrf_fuse([list_a, list_b], fusion_constant=60, top_k=2)
    # Both x and y appear at ranks 1 and 2, so both get the same total score.
    assert out[0]["score"] == pytest.approx(1.0 / 61 + 1.0 / 62)
    assert out[1]["score"] == pytest.approx(1.0 / 61 + 1.0 / 62)


def test_different_overlap_promotes_higher_rank():
    # x is rank 1 in both; y is rank 2 in both. x should win.
    list_a = [_chunk("x"), _chunk("y")]
    list_b = [_chunk("x"), _chunk("y")]
    out = rrf_fuse([list_a, list_b], fusion_constant=60, top_k=2)
    assert out[0]["chunk_id"] == "x"
    assert out[1]["chunk_id"] == "y"


def test_top_k_limits_output():
    ranked = [_chunk(f"c{i}") for i in range(10)]
    out = rrf_fuse([ranked], fusion_constant=60, top_k=3)
    assert len(out) == 3


def test_preserves_source_size_metadata():
    ranked = [_chunk("a", source_size=256)]
    out = rrf_fuse([ranked], fusion_constant=60, top_k=1)
    assert out[0]["source_size"] == 256


def test_fusion_constant_affects_score_magnitude():
    ranked = [_chunk("a")]
    out_low = rrf_fuse([ranked], fusion_constant=1, top_k=1)
    out_high = rrf_fuse([ranked], fusion_constant=60, top_k=1)
    # Smaller fusion_constant produces a larger score for the same rank.
    assert out_low[0]["score"] > out_high[0]["score"]
