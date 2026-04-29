from __future__ import annotations

from typing import Any

from .config import QAConfig
from .logging import get_logger
from .metrics import token_f1
from .qa_gen import _call_openai, _neighbors, _parse_response, _render_answer_prompt
from .types import Chunk, QAPair

log = get_logger(__name__)


def compute_deficits(
    quota: dict[str, int],
    validated_counts: dict[str, int],
    raw_unfiltered_counts: dict[str, int],
) -> dict[str, int]:
    """How many new pairs to generate per type to (optimistically) hit quota.

    Counts both already-validated pairs and pairs sitting unfiltered in
    qa_raw.jsonl toward the quota — the latter on the assumption they will
    pass the filter. If they don't, the next top-up round catches the
    shortfall.
    """
    return {
        t: max(
            0,
            quota[t] - validated_counts.get(t, 0) - raw_unfiltered_counts.get(t, 0),
        )
        for t in quota
    }


def _render_judge_prompt(
    template: str,
    *,
    claimed_type: str,
    question: str,
    answer: str,
    primary: str,
    neighbors: list[str],
) -> str:
    neighbor_block = "\n\n---\n\n".join(neighbors) if neighbors else "(none)"
    return template.format(
        claimed_type=claimed_type,
        question=question,
        answer=answer,
        primary_chunk=primary,
        neighbor_chunks=neighbor_block,
    )


def filter_qa(
    pairs: list[QAPair],
    chunks: list[Chunk],
    qa_cfg: QAConfig,
    answer_prompt: str,
    judge_prompt: str,
) -> tuple[list[QAPair], list[dict[str, Any]]]:
    """Apply the two-filter pipeline to raw QA pairs.

    1. For multihop/synthesis: re-ask the answer model with the primary chunk
       only. If token F1(primary_only_answer, gold) >= threshold, drop it.
    2. For all surviving pairs: ask the judge to keep or drop. The judge sees
       question, gold answer, primary chunk, neighbor chunks, claimed type.

    Returns ``(kept, rejected)`` where kept pairs have ``validated=True`` and
    rejected records carry ``qa_id``, ``type``, ``filter`` (one of
    "primary_only" or "judge"), and ``reason``.
    """
    gen_cfg = qa_cfg.generation
    chunk_index = {c["chunk_id"]: i for i, c in enumerate(chunks)}
    threshold = gen_cfg.primary_only_f1_threshold

    kept: list[QAPair] = []
    rejected: list[dict[str, Any]] = []

    for pair in pairs:
        qtype = pair["type"]
        chunk_id = pair["source_chunk_id"]
        idx = chunk_index.get(chunk_id)
        if idx is None:
            rejected.append(
                {
                    "qa_id": pair["qa_id"],
                    "type": qtype,
                    "filter": "primary_only",
                    "reason": f"source_chunk_id {chunk_id!r} not found in chunks",
                }
            )
            continue

        primary = chunks[idx]
        neighbors = (
            _neighbors(chunks, idx, qa_cfg.neighbor_window) if qtype != "factoid" else []
        )

        if qtype in ("multihop", "synthesis"):
            primary_only_prompt = _render_answer_prompt(
                answer_prompt,
                question_type=qtype,
                question=pair["question"],
                primary=primary["text"],
                neighbors=[],
            )
            raw = _call_openai(
                primary_only_prompt,
                gen_cfg.models.answer,
                gen_cfg,
                gen_cfg.answer_temperature,
            )
            parsed = _parse_response(raw)
            primary_only_answer = ""
            if parsed and not parsed.get("skip"):
                primary_only_answer = str(parsed.get("answer") or "").strip()

            if primary_only_answer:
                f1 = token_f1(primary_only_answer, pair["answer"])
                if f1 >= threshold:
                    rejected.append(
                        {
                            "qa_id": pair["qa_id"],
                            "type": qtype,
                            "filter": "primary_only",
                            "reason": f"primary-only F1={f1:.2f} >= {threshold}",
                        }
                    )
                    continue

        judge_rendered = _render_judge_prompt(
            judge_prompt,
            claimed_type=qtype,
            question=pair["question"],
            answer=pair["answer"],
            primary=primary["text"],
            neighbors=neighbors,
        )
        judge_raw = _call_openai(
            judge_rendered,
            gen_cfg.models.judge,
            gen_cfg,
            gen_cfg.judge_temperature,
        )
        judge_parsed = _parse_response(judge_raw)
        if judge_parsed is None:
            rejected.append(
                {
                    "qa_id": pair["qa_id"],
                    "type": qtype,
                    "filter": "judge",
                    "reason": "judge returned unparseable response",
                }
            )
            continue
        if not judge_parsed.get("keep"):
            rejected.append(
                {
                    "qa_id": pair["qa_id"],
                    "type": qtype,
                    "filter": "judge",
                    "reason": str(judge_parsed.get("reason") or "judge dropped"),
                }
            )
            continue

        kept_pair: QAPair = {**pair, "validated": True}
        kept.append(kept_pair)

    log.info(
        "QA filter done: kept=%d rejected=%d (primary_only=%d, judge=%d)",
        len(kept),
        len(rejected),
        sum(1 for r in rejected if r["filter"] == "primary_only"),
        sum(1 for r in rejected if r["filter"] == "judge"),
    )
    return kept, rejected
