# TODO: Implement text-to-vector embedding using the configured sentence-transformer model.

from __future__ import annotations

import numpy as np


def embed_texts(
    texts: list[str],
    model_name: str,
    device: str,
    batch_size: int,
    normalize: bool,
) -> np.ndarray:
    """Embed a batch of texts and return a 2D array of shape (len(texts), dim)."""
    raise NotImplementedError
