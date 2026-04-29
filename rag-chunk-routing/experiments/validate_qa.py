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
    get_qa_rejected_path,
    get_qa_validated_path,
    read_jsonl,
    write_jsonl,
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


def _load_decided_ids(*paths: Path) -> set[str]:
    decided: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            qa_id = row.get("qa_id")
            if isinstance(qa_id, str):
                decided.add(qa_id)
    return decided


def _append_row(path: Path, row: dict) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _review_pair(
    pair: QAPair,
    primary_text: str,
    neighbors: list[tuple[int, str, str]],
    index: int,
    total: int,
) -> QAPair | None:
    """Return the pair to save (validated=True), or None if rejected.

    ``neighbors`` is a list of (offset, chunk_id, text) tuples, where offset
    is the signed index distance from the primary chunk (e.g. -1 for the
    immediately preceding chunk). They are shown alongside the primary chunk
    so reviewers see the same context the generator was given.
    """
    while True:
        print()
        print(_hr(f"{index}/{total}  type={pair['type']}  chunk={pair['source_chunk_id']}"))
        for offset, nb_id, nb_text in sorted(neighbors):
            if offset >= 0:
                continue
            print(_hr(f"NEIGHBOR {offset:+d}  {nb_id}"))
            print(_wrap(nb_text))
        print(_hr(f"PRIMARY CHUNK  {pair['source_chunk_id']}"))
        print(_wrap(primary_text))
        for offset, nb_id, nb_text in sorted(neighbors):
            if offset <= 0:
                continue
            print(_hr(f"NEIGHBOR {offset:+d}  {nb_id}"))
            print(_wrap(nb_text))
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
    rejected_path = get_qa_rejected_path(config.paths.artifacts_dir)
    chunks_path = get_chunks_path(config.paths.artifacts_dir, config.qa.source_chunk_size)

    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunks file missing: {chunks_path}. Run `make indices` first."
        )
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw QA file missing: {raw_path}. Run `python experiments/generate_qa.py` "
            "and `python experiments/filter_qa.py` first."
        )

    chunks = cast(list[Chunk], read_jsonl(chunks_path))
    chunk_idx_by_id = {c["chunk_id"]: i for i, c in enumerate(chunks)}
    neighbor_window = config.qa.neighbor_window

    raw_pairs = cast(list[QAPair], read_jsonl(raw_path))
    validated_entries = (
        cast(list[QAPair], read_jsonl(validated_path)) if validated_path.exists() else []
    )
    validated_ids = {p["qa_id"] for p in validated_entries}

    pending = list(raw_pairs)

    print(_hr("QA HUMAN REVIEW"))
    print(f"Pending in {raw_path.name}: {len(pending)}")
    judge_passed = sum(1 for p in pending if p["qa_id"] in validated_ids)
    print(
        f"  judge-validated (in {validated_path.name}): {judge_passed}\n"
        f"  not yet judge-validated:                {len(pending) - judge_passed}"
    )
    if not pending:
        print("\nNothing to review.")
        return
    print(
        "\nControls: a=accept  r=reject  e=edit  q=quit & save\n"
        f"Accept → remove from {raw_path.name}; entry stays in {validated_path.name}.\n"
        f"Reject → remove from {raw_path.name} and {validated_path.name}; "
        f"qa_id appended to {rejected_path.name}.\n"
        f"Edit   → remove from {raw_path.name}; updated entry written to {validated_path.name}.\n"
    )

    decided_ids: set[str] = set()
    edits_by_id: dict[str, QAPair] = {}
    rejected_now: set[str] = set()
    accepted_unchanged = 0
    edited = 0
    rejected_count = 0
    try:
        for i, pair in enumerate(pending, start=1):
            primary_idx = chunk_idx_by_id.get(pair["source_chunk_id"])
            if primary_idx is None:
                primary_text = "(chunk text not found)"
                neighbors: list[tuple[int, str, str]] = []
            else:
                primary_text = chunks[primary_idx]["text"]
                neighbors = []
                if pair["type"] != "factoid" and neighbor_window > 0:
                    lo = max(0, primary_idx - neighbor_window)
                    hi = min(len(chunks), primary_idx + neighbor_window + 1)
                    for j in range(lo, hi):
                        if j == primary_idx:
                            continue
                        neighbors.append(
                            (j - primary_idx, chunks[j]["chunk_id"], chunks[j]["text"])
                        )
            result = _review_pair(pair, primary_text, neighbors, index=i, total=len(pending))
            decided_ids.add(pair["qa_id"])
            if result is None:
                _append_row(rejected_path, {"qa_id": pair["qa_id"]})
                rejected_now.add(pair["qa_id"])
                rejected_count += 1
            else:
                if result["question"] != pair["question"] or result["answer"] != pair["answer"]:
                    edits_by_id[pair["qa_id"]] = result
                    edited += 1
                else:
                    if pair["qa_id"] not in validated_ids:
                        # First-pass accept of a pair the judge never validated.
                        edits_by_id[pair["qa_id"]] = cast(
                            QAPair, {**pair, "validated": True}
                        )
                    accepted_unchanged += 1
    except KeyboardInterrupt:
        print("\nStopping early — progress saved.")

    # Rewrite validated: drop rejected entries; replace edits; preserve everything else.
    if decided_ids:
        merged: list[QAPair] = []
        seen: set[str] = set()
        for p in validated_entries:
            qid = p["qa_id"]
            if qid in seen:
                continue
            if qid in rejected_now:
                seen.add(qid)
                continue
            if qid in edits_by_id:
                merged.append(edits_by_id[qid])
                seen.add(qid)
                continue
            merged.append(p)
            seen.add(qid)
        # Add accepted-with-edits or first-pass accepts that weren't previously in validated.
        for qid, entry in edits_by_id.items():
            if qid not in seen:
                merged.append(entry)
                seen.add(qid)
        write_jsonl(validated_path, [dict(p) for p in merged])

        # Remove decided pairs from qa_raw.jsonl.
        remaining_raw = [p for p in raw_pairs if p["qa_id"] not in decided_ids]
        write_jsonl(raw_path, [dict(p) for p in remaining_raw])

    print()
    print(_hr("SUMMARY"))
    print(f"Accepted (unchanged): {accepted_unchanged}")
    print(f"Edited (kept):        {edited}")
    print(f"Rejected:             {rejected_count}")
    print(f"Removed from raw:     {len(decided_ids)}")
    print(f"Validated file: {validated_path}")
    print(f"Rejected file:  {rejected_path}")
    print(f"Raw file:       {raw_path}")


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
