# TODO: Provide JSONL read/write helpers and path resolution for artifacts and results. No data transformations belong here.

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_text(path: str | Path) -> str:
    path = Path(path)
    return path.read_text(encoding="utf-8")


def write_text(path: str | Path, content: str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")


def read_json(path: str | Path) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any, indent: int = 2) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
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
    path = Path(path)
    ensure_dir(path.parent)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_chunks_path(artifacts_dir: str | Path, size: int) -> Path:
    return Path(artifacts_dir) / "chunks" / f"{size}.jsonl"


def get_index_path(artifacts_dir: str | Path, size: int) -> Path:
    return Path(artifacts_dir) / "indices" / f"{size}.faiss"


def get_qa_raw_path(artifacts_dir: str | Path) -> Path:
    return Path(artifacts_dir) / "qa" / "qa_raw.jsonl"


def get_qa_validated_path(artifacts_dir: str | Path) -> Path:
    return Path(artifacts_dir) / "qa" / "qa_validated.jsonl"


def get_oracle_labels_path(artifacts_dir: str | Path) -> Path:
    return Path(artifacts_dir) / "qa" / "oracle_labels.jsonl"


def create_run_dir(results_dir: str | Path, system_name: str) -> Path:
    run_dir = Path(results_dir) / "runs" / f"{timestamp()}_{system_name}"
    ensure_dir(run_dir)
    return run_dir