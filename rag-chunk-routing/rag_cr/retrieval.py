from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import numpy as np

from .config import Config
from .embedding import embed_texts
from .indexing import search_index
from .io import get_chunks_path, get_index_path, read_jsonl
from .types import Chunk, RetrievedChunk

ChunkScale = int | Literal["fusion"]


class Retriever:
    """Unified retriever over all configured chunk sizes.

    One instance loads every per-size index once and then serves both
    single-scale queries (``chunk_size=256``) and fused multi-scale queries
    (``chunk_size='fusion'``) through the same ``retrieve`` method.
    """

    def __init__(self, config: Config, artifacts_dir: str | Path | None = None) -> None:
        self.config = config
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else config.paths.artifacts_dir
        self._chunks: dict[int, list[Chunk]] = {}

    def _load_chunks(self, size: int) -> list[Chunk]:
        if size not in self._chunks:
            rows = read_jsonl(get_chunks_path(self.artifacts_dir, size))
            self._chunks[size] = [cast(Chunk, r) for r in rows]
        return self._chunks[size]

    def _embed_query(self, query: str) -> np.ndarray:
        cfg = self.config.embedding
        return embed_texts([query], cfg.model_name, cfg.device, 1, cfg.normalize)

    def retrieve(self, query: str, chunk_size: ChunkScale, k: int) -> list[RetrievedChunk]:
        """Return the top-k chunks for a query from one index or from fusion."""
        query_emb = self._embed_query(query)
        if chunk_size == "fusion":
            return self._retrieve_fusion(query_emb, k)
        return self._retrieve_single(query_emb, chunk_size, k)

    def _retrieve_single(self, query_emb: np.ndarray, size: int, k: int) -> list[RetrievedChunk]:
        scores, indices = search_index(get_index_path(self.artifacts_dir, size), query_emb, k)
        chunks = self._load_chunks(size)
        result: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            c = chunks[int(idx)]
            result.append(
                RetrievedChunk(chunk_id=c["chunk_id"], text=c["text"], score=float(score), source_size=size)
            )
        return result

    def _retrieve_fusion(self, query_emb: np.ndarray, k: int) -> list[RetrievedChunk]:
        fusion_k = self.config.retrieval.fusion_constant
        scored: list[tuple[float, RetrievedChunk]] = []
        for size in self.config.chunking.sizes:
            scores, indices = search_index(get_index_path(self.artifacts_dir, size), query_emb, k)
            chunks = self._load_chunks(size)
            for rank, idx in enumerate(indices[0], start=1):
                if idx < 0:
                    continue
                c = chunks[int(idx)]
                rrf = 1.0 / (fusion_k + rank)
                scored.append(
                    (rrf, RetrievedChunk(chunk_id=c["chunk_id"], text=c["text"], score=rrf, source_size=size))
                )
        scored.sort(key=lambda x: x[0], reverse=True)
        return [rc for _, rc in scored[:k]]
