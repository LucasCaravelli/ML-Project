"""BGE-small dense embedding of chunks and queries via HuggingFace transformers."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=4)
def _load_model(model_name: str, device: str) -> SentenceTransformer:
    return SentenceTransformer(model_name, device=device)


def embed_texts(
    texts: list[str],
    model_name: str,
    device: str,
    batch_size: int,
    normalize: bool,
) -> np.ndarray:
    """Embed a batch of texts and return a 2D array of shape (len(texts), dim)."""
    model = _load_model(model_name, device)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return np.asarray(embeddings, dtype=np.float32)
