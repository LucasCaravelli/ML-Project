"""JSONL/JSON read+write helpers and standardized artifact/results path resolution."""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Create the directory and all missing parents; return the resolved Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_text(path: str | Path) -> str:
    """Read a UTF-8 text file into a string."""
    path = Path(path)
    return path.read_text(encoding="utf-8")


def write_text(path: str | Path, content: str) -> None:
    """Write a UTF-8 text file, creating parent directories if needed."""
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")


def read_json(path: str | Path) -> Any:
    """Load a JSON file (UTF-8)."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any, indent: int = 2) -> None:
    """Serialize a JSON-compatible object to a file, creating parents if needed."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts; reject malformed or non-object lines."""
    path = Path(path)
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from e

            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_number} of {path} is not a JSON object.")

            rows.append(obj)

    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write a sequence of dicts to a JSONL file, creating parents if needed."""
    path = Path(path)
    ensure_dir(path.parent)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def timestamp() -> str:
    """Return the current local timestamp as ``YYYYMMDD_HHMMSS`` (used in run-dir names)."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_chunks_path(artifacts_dir: str | Path, size: int) -> Path:
    """Path to the chunks JSONL for a given chunk size."""
    return Path(artifacts_dir) / "chunks" / f"{size}.jsonl"


def get_index_path(artifacts_dir: str | Path, size: int) -> Path:
    """Path to the FAISS index file for a given chunk size."""
    return Path(artifacts_dir) / "indices" / f"{size}.faiss"


def get_qa_raw_path(artifacts_dir: str | Path) -> Path:
    """Path to the raw (unfiltered) QA pairs JSONL."""
    return Path(artifacts_dir) / "qa" / "qa_raw.jsonl"


def get_qa_validated_path(artifacts_dir: str | Path) -> Path:
    """Path to the human-validated QA pairs JSONL."""
    return Path(artifacts_dir) / "qa" / "qa_validated.jsonl"


def get_qa_rejected_path(artifacts_dir: str | Path) -> Path:
    """Path to the rejected QA pairs JSONL."""
    return Path(artifacts_dir) / "qa" / "qa_rejected.jsonl"


def get_oracle_labels_path(artifacts_dir: str | Path) -> Path:
    """Path to the oracle-best-size labels JSONL (the router's training target)."""
    return Path(artifacts_dir) / "qa" / "oracle_labels.jsonl"


def create_run_dir(results_dir: str | Path, system_name: str) -> Path:
    """Create and return a timestamped run directory under ``<results>/runs/``."""
    run_dir = Path(results_dir) / "runs" / f"{timestamp()}_{system_name}"
    ensure_dir(run_dir)
    return run_dir