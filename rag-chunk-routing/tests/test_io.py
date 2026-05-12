from __future__ import annotations

from pathlib import Path

import pytest

from rag_cr.io import (
    create_run_dir,
    ensure_dir,
    get_chunks_path,
    get_index_path,
    get_qa_raw_path,
    get_qa_rejected_path,
    get_qa_validated_path,
    read_json,
    read_jsonl,
    read_text,
    write_json,
    write_jsonl,
    write_text,
)


def test_ensure_dir_creates_missing_parents(tmp_path: Path):
    target = tmp_path / "a" / "b" / "c"
    out = ensure_dir(target)
    assert target.is_dir()
    assert out == target


def test_ensure_dir_is_idempotent(tmp_path: Path):
    target = tmp_path / "x"
    ensure_dir(target)
    ensure_dir(target)
    assert target.is_dir()


def test_text_round_trip(tmp_path: Path):
    path = tmp_path / "nested" / "file.txt"
    write_text(path, "hello world\nline two\n")
    assert read_text(path) == "hello world\nline two\n"


def test_json_round_trip(tmp_path: Path):
    path = tmp_path / "out" / "data.json"
    payload = {"a": 1, "b": [2, 3], "c": "x"}
    write_json(path, payload)
    assert read_json(path) == payload


def test_jsonl_round_trip(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    rows = [{"qa_id": "qa_001", "type": "factoid"}, {"qa_id": "qa_002", "type": "multihop"}]
    write_jsonl(path, rows)
    assert read_jsonl(path) == rows


def test_jsonl_skips_blank_lines(tmp_path: Path):
    path = tmp_path / "blanks.jsonl"
    path.write_text('{"a": 1}\n\n   \n{"b": 2}\n', encoding="utf-8")
    assert read_jsonl(path) == [{"a": 1}, {"b": 2}]


def test_jsonl_rejects_invalid_json_with_line_number(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"ok": true}\nnot valid json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        read_jsonl(path)


def test_jsonl_rejects_non_object_line(tmp_path: Path):
    path = tmp_path / "list.jsonl"
    path.write_text('{"ok": true}\n[1, 2, 3]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="(?i)line 2"):
        read_jsonl(path)


def test_path_helpers_compose_correctly(tmp_path: Path):
    assert get_chunks_path(tmp_path, 256) == tmp_path / "chunks" / "256.jsonl"
    assert get_index_path(tmp_path, 256) == tmp_path / "indices" / "256.faiss"
    assert get_qa_raw_path(tmp_path) == tmp_path / "qa" / "qa_raw.jsonl"
    assert get_qa_validated_path(tmp_path) == tmp_path / "qa" / "qa_validated.jsonl"
    assert get_qa_rejected_path(tmp_path) == tmp_path / "qa" / "qa_rejected.jsonl"


def test_create_run_dir_makes_timestamp_dir(tmp_path: Path):
    run_dir = create_run_dir(tmp_path, "fusion")
    assert run_dir.is_dir()
    assert run_dir.parent == tmp_path / "runs"
    assert run_dir.name.endswith("_fusion")
