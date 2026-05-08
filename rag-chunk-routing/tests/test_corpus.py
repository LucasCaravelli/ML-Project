from __future__ import annotations

from pathlib import Path

import pytest

from rag_cr.corpus import load_corpus


def test_load_corpus_reads_text(tmp_path: Path):
    path = tmp_path / "corpus.txt"
    path.write_text("hello corpus", encoding="utf-8")
    assert load_corpus(path) == "hello corpus"


def test_load_corpus_raises_on_missing_file(tmp_path: Path):
    missing = tmp_path / "nope.txt"
    with pytest.raises(FileNotFoundError, match="nope"):
        load_corpus(missing)
