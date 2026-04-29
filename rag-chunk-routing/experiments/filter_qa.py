from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv

from rag_cr import Config, load_config, set_seed
from rag_cr.io import (
    get_chunks_path,
    get_qa_raw_path,
    get_qa_rejected_path,
    get_qa_validated_path,
    read_jsonl,
    read_text,
    write_jsonl,
)
from rag_cr.logging import get_logger
from rag_cr.qa_filter import compute_deficits, filter_qa
from rag_cr.qa_gen import _targets_from_distribution, generate_qa
from rag_cr.types import Chunk, QAPair

log = get_logger(__name__)


def _read_pairs(path: Path) -> list[QAPair]:
    if not path.exists():
        return []
    return cast(list[QAPair], read_jsonl(path))


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path)


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    existing = read_jsonl(path) if path.exists() else []
    write_jsonl(path, existing + rows)


def _count_by_type(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        t = row.get("type")
        if isinstance(t, str):
            counts[t] = counts.get(t, 0) + 1
    return counts


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


def run_pipeline(config: Config, *, do_topup: bool) -> None:
    set_seed(config.project.seed)
    load_dotenv()

    chunks_path = get_chunks_path(config.paths.artifacts_dir, config.qa.source_chunk_size)
    raw_path = get_qa_raw_path(config.paths.artifacts_dir)
    validated_path = get_qa_validated_path(config.paths.artifacts_dir)
    rejected_path = get_qa_rejected_path(config.paths.artifacts_dir)

    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file missing: {chunks_path}. Run `make indices` first.")
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw QA file missing: {raw_path}. Run `python experiments/generate_qa.py` first."
        )

    chunks = cast(list[Chunk], read_jsonl(chunks_path))
    answer_prompt = read_text(config.prompts.qa_answer)
    judge_prompt = read_text(config.prompts.judge)
    quota = _targets_from_distribution(config.qa.target_count, config.qa.type_distribution)
    log.info("Per-type quotas for target_count=%d: %s", config.qa.target_count, quota)

    question_prompt_paths = {
        "factoid": config.prompts.qa_generation_factoid,
        "multihop": config.prompts.qa_generation_multihop,
        "synthesis": config.prompts.qa_generation_synthesis,
    }

    max_rounds = config.qa.generation.max_topup_rounds if do_topup else 1
    for round_idx in range(max_rounds):
        all_raw = _read_pairs(raw_path)
        validated_pairs = _read_pairs(validated_path)
        rejected_records = _read_records(rejected_path)
        decided_ids = {p["qa_id"] for p in validated_pairs} | {
            r.get("qa_id") for r in rejected_records if r.get("qa_id")
        }
        unfiltered = [p for p in all_raw if p["qa_id"] not in decided_ids]

        validated_counts = _count_by_type(cast(list[dict[str, Any]], validated_pairs))
        raw_unfiltered_counts = _count_by_type(cast(list[dict[str, Any]], unfiltered))
        deficits = compute_deficits(quota, validated_counts, raw_unfiltered_counts)
        log.info(
            "Round %d: validated=%s raw_unfiltered=%s deficits=%s",
            round_idx,
            validated_counts,
            raw_unfiltered_counts,
            deficits,
        )

        new_pairs: list[QAPair] = []
        if do_topup and any(d > 0 for d in deficits.values()):
            qa_id_offset = _next_qa_id_offset(raw_path, validated_path, rejected_path)
            validated_chunk_ids = {
                p["source_chunk_id"]
                for p in validated_pairs
                if isinstance(p.get("source_chunk_id"), str)
            }
            for qtype, n in deficits.items():
                if n <= 0:
                    continue
                generated = generate_qa(
                    chunks_path=chunks_path,
                    qa_cfg=config.qa,
                    question_prompt_paths=question_prompt_paths,
                    answer_prompt_path=config.prompts.qa_answer,
                    limit=n,
                    seed=config.project.seed + round_idx,
                    qa_id_offset=qa_id_offset,
                    exclude_chunk_ids=validated_chunk_ids,
                    type_override=qtype,
                )
                qa_id_offset += len(generated)
                new_pairs.extend(generated)
            if new_pairs:
                _append_jsonl(raw_path, cast(list[dict[str, Any]], new_pairs))
                log.info("Generated %d new raw pairs across deficient types", len(new_pairs))

        to_filter = unfiltered + new_pairs
        if not to_filter:
            log.info("No unfiltered pairs and no deficit — stopping.")
            break

        kept, rejected = filter_qa(
            to_filter,
            chunks,
            config.qa,
            answer_prompt=answer_prompt,
            judge_prompt=judge_prompt,
        )
        _append_jsonl(validated_path, cast(list[dict[str, Any]], kept))
        _append_jsonl(rejected_path, rejected)

        # Filter-rejected pairs are terminal — drop them from qa_raw.jsonl so
        # the human-review pool only contains judge-accepted pairs.
        rejected_ids = {r["qa_id"] for r in rejected if r.get("qa_id")}
        if rejected_ids and raw_path.exists():
            current_raw = _read_pairs(raw_path)
            remaining_raw = [p for p in current_raw if p["qa_id"] not in rejected_ids]
            if len(remaining_raw) != len(current_raw):
                write_jsonl(raw_path, [dict(p) for p in remaining_raw])
        log.info(
            "Round %d done: kept=%d rejected=%d (rejected dropped from raw)",
            round_idx,
            len(kept),
            len(rejected),
        )

        post_validated = _count_by_type(cast(list[dict[str, Any]], _read_pairs(validated_path)))
        if all(post_validated.get(t, 0) >= quota[t] for t in quota):
            log.info("All per-type quotas met. Stopping at round %d.", round_idx)
            break

    log.info(
        "Final per-type validated counts: %s",
        _count_by_type(cast(list[dict[str, Any]], _read_pairs(validated_path))),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument(
        "--no-topup",
        action="store_true",
        help="Run a single filter pass over qa_raw.jsonl without generating top-up pairs.",
    )
    args = parser.parse_args()
    run_pipeline(load_config(args.config), do_topup=not args.no_topup)
