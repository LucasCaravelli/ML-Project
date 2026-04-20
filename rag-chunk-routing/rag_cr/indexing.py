# TODO: Implement FAISS index construction, persistence, and search for a given chunk size.

from __future__ import annotations

from pathlib import Path

import numpy as np


def build_index(embeddings: np.ndarray, index_path: Path) -> None:
    """Build a FAISS index over the embeddings and persist it to disk."""
    raise NotImplementedError


def search_index(
    index_path: Path,
    query_embeddings: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Search a persisted index and return (scores, indices) arrays."""
    raise NotImplementedError
