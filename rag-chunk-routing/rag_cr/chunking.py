"""Deterministic tokenizer-aware chunking of the corpus into fixed-size segments."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .types import Chunk

# Must match embedding.model_name in configs/base.yaml — same tokenizer everywhere.
_TOKENIZER_MODEL = "BAAI/bge-small-en-v1.5"


@lru_cache(maxsize=1)
def _tokenizer() -> PreTrainedTokenizerBase:
    import logging
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    return AutoTokenizer.from_pretrained(_TOKENIZER_MODEL)


def chunk_corpus(corpus_path: Path, size: int, overlap: int) -> list[Chunk]:
    """Split the corpus into deterministic chunks of the target size."""
    text = Path(corpus_path).read_text(encoding="utf-8")
    tok = _tokenizer()
    encoding = tok(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    offsets: list[tuple[int, int]] = encoding["offset_mapping"]
    n_tokens = len(offsets)
    # Stride is how far the window advances per chunk. With overlap > 0 each
    # chunk shares ``overlap`` tokens with the previous one, which guards
    # against entity/sentence boundaries falling exactly at a chunk seam.
    stride = size - overlap

    chunks: list[Chunk] = []
    tok_start = 0
    chunk_idx = 0

    while tok_start < n_tokens:
        # Last chunk may end short of ``size`` tokens (the corpus tail). The
        # ``size`` field below still records the *target* size — that's how
        # retrieval/fusion code keys per-size indices, regardless of the
        # actual length of the trailing chunk.
        tok_end = min(tok_start + size, n_tokens)
        start_char = offsets[tok_start][0]
        end_char = offsets[tok_end - 1][1]
        chunks.append(
            Chunk(
                chunk_id=f"{size}-{chunk_idx:06d}",
                size=size,
                start_char=start_char,
                end_char=end_char,
                text=text[start_char:end_char],
            )
        )
        chunk_idx += 1
        tok_start += stride

    return chunks
