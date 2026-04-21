from __future__ import annotations

import argparse
import json
import shutil
import sys
import textwrap
from pathlib import Path
from typing import cast

from rag_cr import Config, load_config, set_seed
from rag_cr.io import (
    ensure_dir,
    get_chunks_path,
    get_qa_raw_path,
    get_qa_validated_path,
    read_jsonl,
)
from rag_cr.types import Chunk, QAPair


RULE = "─"


def _term_width(default: int = 80) -> int:
    try:
        return max(40, shutil.get_terminal_size().columns)
    except OSError:
        return default


def _hr(label: str = "") -> str:
    width = _term_width()
    if not label:
        return RULE * width
    pad = max(0, width - len(label) - 4)
    left = pad // 2
    right = pad - left
    return f"{RULE * left} {label} {RULE * right}"


def _wrap(text: str) -> str:
    width = _term_width()
    return "\n".join(
        textwrap.fill(line, width=width, replace_whitespace=False, drop_whitespace=False)
        if line.strip()
        else line
        for line in text.splitlines()
    )


def _prompt(msg: str) -> str:
    try:
        return input(msg)
    except EOFError:
        return "q"


def _edit_field(label: str, current: str) -> str:
    print(f"\nCurrent {label}:")
    print(_wrap(current))
    new = _prompt(f"\nNew {label} (leave blank to keep): ").strip()
    return new if new else current


def _load_decided_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    decided: set[str] = set()
    for row in read_jsonl(path):
        qa_id = row.get("qa_id")
        if isinstance(qa_id, str):
            decided.add(qa_id)
    return decided


def _append_pair(path: Path, pair: QAPair) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(pair, ensure_ascii=False) + "\n")


def _review_pair(
    pair: QAPair,
    chunk_text: str,
    index: int,
    total: int,
) -> QAPair | None:
    """Return the pair to save (validated=True), or None if rejected."""
    while True:
        print()
        print(_hr(f"{index}/{total}  type={pair['type']}  chunk={pair['source_chunk_id']}"))
        print(_hr("CHUNK"))
        print(_wrap(chunk_text))
        print(_hr("QUESTION"))
        print(_wrap(pair["question"]))
        print(_hr("ANSWER"))
        print(_wrap(pair["answer"]))
        print(_hr())
        choice = _prompt("[a]ccept  [r]eject  [e]dit  [q]uit & save  > ").strip().lower()

        if choice in ("a", "accept"):
            return cast(QAPair, {**pair, "validated": True})
        if choice in ("r", "reject"):
            return None
        if choice in ("e", "edit"):
            new_q = _edit_field("question", pair["question"])
            new_a = _edit_field("answer", pair["answer"])
            return cast(
                QAPair,
                {**pair, "question": new_q, "answer": new_a, "validated": True},
            )
        if choice in ("q", "quit"):
            raise KeyboardInterrupt
        print("Unrecognized choice. Use a / r / e / q.")


def main(config: Config) -> None:
    set_seed(config.project.seed)

    raw_path = get_qa_raw_path(config.paths.artifacts_dir)
    validated_path = get_qa_validated_path(config.paths.artifacts_dir)
    chunks_path = get_chunks_path(config.paths.artifacts_dir, config.qa.source_chunk_size)

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw QA file missing: {raw_path}. Run `python experiments/generate_qa.py` first."
        )
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunks file missing: {chunks_path}. Run `make indices` first."
        )

    raw_pairs = cast(list[QAPair], read_jsonl(raw_path))
    chunks = cast(list[Chunk], read_jsonl(chunks_path))
    chunk_text_by_id = {c["chunk_id"]: c["text"] for c in chunks}

    decided = _load_decided_ids(validated_path)
    pending = [p for p in raw_pairs if p["qa_id"] not in decided]

    print(_hr("QA VALIDATION"))
    print(f"Raw candidates:    {len(raw_pairs)}")
    print(f"Already decided:   {len(decided)}  (kept in {validated_path.name})")
    print(f"Pending this run:  {len(pending)}")
    if not pending:
        print("\nNothing to review. All raw candidates already have a decision.")
        return
    print("\nControls: a=accept  r=reject  e=edit  q=quit & save")
    print("Rejected pairs are dropped. Edits are saved as accepted.\n")

    accepted = 0
    rejected = 0
    try:
        for i, pair in enumerate(pending, start=1):
            chunk_text = chunk_text_by_id.get(
                pair["source_chunk_id"], "(chunk text not found)"
            )
            result = _review_pair(pair, chunk_text, index=i, total=len(pending))
            if result is None:
                rejected += 1
            else:
                _append_pair(validated_path, result)
                accepted += 1
    except KeyboardInterrupt:
        print("\nStopping early — progress saved.")

    print()
    print(_hr("SUMMARY"))
    print(f"Accepted: {accepted}")
    print(f"Rejected: {rejected}")
    print(f"Validated file: {validated_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interactively review raw QA candidates and write validated pairs."
    )
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    try:
        main(load_config(args.config))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
