"""FAISS index construction, persistence, and search per chunk size."""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np

from .io import ensure_dir

# Module-level cache so repeated searches in the same process skip disk reads.
_index_cache: dict[str, faiss.Index] = {}


def _load_index(index_path: Path) -> faiss.Index:
    key = str(index_path)
    if key not in _index_cache:
        _index_cache[key] = faiss.read_index(key)
    return _index_cache[key]


def build_index(embeddings: np.ndarray, index_path: Path) -> None:
    """Build a FAISS flat inner-product index and persist it to disk."""
    index_path = Path(index_path)
    vectors = np.asarray(embeddings, dtype=np.float32)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    ensure_dir(index_path.parent)
    faiss.write_index(index, str(index_path))
    _index_cache[str(index_path)] = index


def search_index(
    index_path: Path,
    query_embeddings: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Search a persisted index and return (scores, indices) arrays."""
    index = _load_index(Path(index_path))
    scores, indices = index.search(np.asarray(query_embeddings, dtype=np.float32), k)
    return scores, indices
