"""Synthesize raw QA pairs from the corpus chunks via the OpenAI API.

Loads the primary-size chunks, allocates collision-free qa_ids that respect
existing raw / validated / rejected sets, and produces type-stratified QA
pairs (factoid / multihop / synthesis) appended to artifacts/qa/qa_raw.jsonl.
Idempotent across runs unless --force is set.

Run from rag-chunk-routing/:
    python experiments/generate_qa.py --config configs/base.yaml [--limit N] [--force]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from rag_cr import Config, load_config, set_seed
from rag_cr.io import (
    get_chunks_path,
    get_qa_raw_path,
    get_qa_rejected_path,
    get_qa_validated_path,
    read_jsonl,
    write_jsonl,
)
from rag_cr.logging import get_logger
from rag_cr.qa_filter import compute_deficits
from rag_cr.qa_gen import _targets_from_distribution, generate_qa
from rag_cr.types import QAPair

log = get_logger(__name__)


def _next_qa_id_offset(*paths: Path) -> int:
    max_idx = -1
    for path in paths:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            qa_id = row.get("qa_id", "")
            if not (isinstance(qa_id, str) and qa_id.startswith("qa_")):
                continue
            try:
                idx = int(qa_id[3:])
            except ValueError:
                continue
            if idx > max_idx:
                max_idx = idx
    return max_idx + 1


def _all_qa_ids(*paths: Path) -> set[str]:
    """Collect every qa_id already on disk so generation can skip collisions."""
    ids: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            qa_id = row.get("qa_id")
            if isinstance(qa_id, str) and qa_id.startswith("qa_"):
                ids.add(qa_id)
    return ids


def _validated_chunk_ids(validated_path: Path) -> set[str]:
    if not validated_path.exists():
        return set()
    return {
        row["source_chunk_id"]
        for row in read_jsonl(validated_path)
        if isinstance(row.get("source_chunk_id"), str)
    }


def _count_by_type(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    counts: dict[str, int] = {}
    for row in read_jsonl(path):
        t = row.get("type")
        if isinstance(t, str):
            counts[t] = counts.get(t, 0) + 1
    return counts


def main(config: Config, limit: int | None, force: bool) -> None:
    """Synthesize raw QA pairs and append them to artifacts/qa/qa_raw.jsonl."""
    set_seed(config.project.seed)
    load_dotenv()

    chunks_path = get_chunks_path(config.paths.artifacts_dir, config.qa.source_chunk_size)
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunks file missing: {chunks_path}. Run `make indices` first."
        )

    out_path = get_qa_raw_path(config.paths.artifacts_dir)
    validated_path = get_qa_validated_path(config.paths.artifacts_dir)
    rejected_path = get_qa_rejected_path(config.paths.artifacts_dir)

    if force and out_path.exists():
        existing_raw: list[QAPair] = []
    else:
        existing_raw = read_jsonl(out_path) if out_path.exists() else []  # type: ignore[assignment]

    exclude = _validated_chunk_ids(validated_path)
    qa_id_offset = _next_qa_id_offset(out_path, validated_path, rejected_path)
    existing_qa_ids = _all_qa_ids(out_path, validated_path, rejected_path)

    quota = _targets_from_distribution(config.qa.target_count, config.qa.type_distribution)
    validated_counts = _count_by_type(validated_path)
    raw_unfiltered_counts = _count_by_type(out_path) if not force else {}
    deficits = compute_deficits(quota, validated_counts, raw_unfiltered_counts)

    if limit is not None:
        # Cap per-type generation evenly within the limit.
        cap = _targets_from_distribution(limit, config.qa.type_distribution)
        deficits = {t: min(deficits.get(t, 0), cap.get(t, 0)) for t in quota}

    total_to_generate = sum(deficits.values())
    log.info(
        "Per-type quota=%s validated=%s raw_unfiltered=%s -> generating %s (total=%d)",
        quota,
        validated_counts,
        raw_unfiltered_counts,
        deficits,
        total_to_generate,
    )

    if total_to_generate == 0:
        log.info("All per-type quotas already met. Nothing to generate.")
        return

    question_prompt_paths = {
        "factoid": config.prompts.qa_generation_factoid,
        "multihop": config.prompts.qa_generation_multihop,
        "synthesis": config.prompts.qa_generation_synthesis,
    }

    new_pairs: list[QAPair] = []
    for qtype, n in deficits.items():
        if n <= 0:
            continue
        generated = generate_qa(
            chunks_path=chunks_path,
            qa_cfg=config.qa,
            question_prompt_paths=question_prompt_paths,
            answer_prompt_path=config.prompts.qa_answer,
            limit=n,
            seed=config.project.seed,
            qa_id_offset=qa_id_offset,
            exclude_chunk_ids=exclude,
            type_override=qtype,
            existing_qa_ids=existing_qa_ids,
        )
        for p in generated:
            existing_qa_ids.add(p["qa_id"])
            idx = int(p["qa_id"][3:])
            if idx + 1 > qa_id_offset:
                qa_id_offset = idx + 1
        new_pairs.extend(generated)

    if force:
        write_jsonl(out_path, new_pairs)
        log.info("Wrote %d new raw QA pairs to %s (overwrote)", len(new_pairs), out_path)
    else:
        write_jsonl(out_path, existing_raw + new_pairs)
        log.info(
            "Appended %d new raw QA pairs to %s (now %d total unfiltered)",
            len(new_pairs),
            out_path,
            len(existing_raw) + len(new_pairs),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Override qa.initial_batch_size for small test runs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing non-empty artifacts/qa/qa_raw.jsonl.",
    )
    args = parser.parse_args()
    main(load_config(args.config), limit=args.limit, force=args.force)
