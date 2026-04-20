# TODO: Implement the unified retrieval interface for both single-scale retrieval and multi-scale reciprocal-rank-fusion retrieval.

from __future__ import annotations

from pathlib import Path
from typing import Literal

from .config import Config
from .types import RetrievedChunk

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

    def retrieve(self, query: str, chunk_size: ChunkScale, k: int) -> list[RetrievedChunk]:
        """Return the top-k chunks for a query from one index or from fusion."""
        raise NotImplementedError
