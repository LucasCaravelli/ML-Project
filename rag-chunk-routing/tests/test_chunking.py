from __future__ import annotations

from pathlib import Path

import pytest

from rag_cr.chunking import chunk_corpus


@pytest.fixture
def corpus_file(tmp_path: Path) -> Path:
    text = (
        "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "
        "nu xi omicron pi rho sigma tau upsilon phi chi psi omega. "
        "The quick brown fox jumps over the lazy dog. "
        "Pack my box with five dozen liquor jugs. "
        "How vexingly quick daft zebras jump. "
        "Sphinx of black quartz judge my vow. "
    ) * 10
    path = tmp_path / "corpus.txt"
    path.write_text(text, encoding="utf-8")
    return path


def test_chunking_is_deterministic(corpus_file: Path):
    a = chunk_corpus(corpus_file, size=64, overlap=8)
    b = chunk_corpus(corpus_file, size=64, overlap=8)
    assert len(a) == len(b)
    for ca, cb in zip(a, b, strict=True):
        assert ca == cb


def test_chunk_ids_are_sequential_and_unique(corpus_file: Path):
    chunks = chunk_corpus(corpus_file, size=64, overlap=8)
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))
    for idx, chunk in enumerate(chunks):
        assert chunk["chunk_id"] == f"64-{idx:06d}"


def test_chunk_size_field_matches_target(corpus_file: Path):
    for size in (32, 64, 128):
        chunks = chunk_corpus(corpus_file, size=size, overlap=4)
        assert all(c["size"] == size for c in chunks)


def test_first_chunk_starts_at_zero(corpus_file: Path):
    chunks = chunk_corpus(corpus_file, size=64, overlap=8)
    assert chunks[0]["start_char"] == 0


def test_chunk_text_matches_offsets(corpus_file: Path):
    text = corpus_file.read_text(encoding="utf-8")
    chunks = chunk_corpus(corpus_file, size=64, overlap=8)
    for chunk in chunks:
        assert chunk["text"] == text[chunk["start_char"]:chunk["end_char"]]


def test_overlap_yields_at_least_as_many_chunks(corpus_file: Path):
    no_overlap = chunk_corpus(corpus_file, size=64, overlap=0)
    with_overlap = chunk_corpus(corpus_file, size=64, overlap=32)
    assert len(with_overlap) >= len(no_overlap)


def test_different_target_sizes_produce_different_chunkings(corpus_file: Path):
    small = chunk_corpus(corpus_file, size=32, overlap=0)
    large = chunk_corpus(corpus_file, size=128, overlap=0)
    assert len(small) > len(large)
