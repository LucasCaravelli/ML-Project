"""Corpus I/O and metadata accessors."""

from __future__ import annotations

from pathlib import Path


def load_corpus(corpus_path: str | Path) -> str:
    """Read the raw corpus text from disk."""
    corpus_path = Path(corpus_path)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found at {corpus_path}")
    return corpus_path.read_text(encoding="utf-8")
