# TODO: Implement chunking logic that splits the corpus into chunks at the configured sizes, using a tokenizer consistent with the embedding model.

from __future__ import annotations

from pathlib import Path

from .types import Chunk


def chunk_corpus(corpus_path: Path, size: int, overlap: int) -> list[Chunk]:
    """Split the corpus into deterministic chunks of the target size."""
    raise NotImplementedError
