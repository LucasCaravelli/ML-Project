from __future__ import annotations

from .types import RetrievedChunk


def rrf_fuse(
    ranked_per_size: list[list[RetrievedChunk]],
    fusion_constant: int,
    top_k: int,
) -> list[RetrievedChunk]:
    """Reciprocal Rank Fusion: score(d) = Σ_i 1/(fusion_constant + rank_i(d)).

    Parameters are required (no defaults) so callers must pass values from
    config.retrieval.fusion_constant and config.retrieval.top_k explicitly.
    """
    acc: dict[str, tuple[float, RetrievedChunk]] = {}
    for ranked in ranked_per_size:
        for rank, chunk in enumerate(ranked, start=1):
            cid = chunk["chunk_id"]
            rrf = 1.0 / (fusion_constant + rank)
            if cid in acc:
                prev_score, rc = acc[cid]
                acc[cid] = (prev_score + rrf, rc)
            else:
                acc[cid] = (rrf, chunk)
    sorted_items = sorted(acc.values(), key=lambda x: x[0], reverse=True)[:top_k]
    return [
        RetrievedChunk(chunk_id=rc["chunk_id"], text=rc["text"], score=sc, source_size=rc["source_size"])
        for sc, rc in sorted_items
    ]
