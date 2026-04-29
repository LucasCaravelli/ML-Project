from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import cast

from .config import QAConfig, QAGenerationConfig
from .io import read_jsonl, read_text
from .logging import get_logger
from .types import Chunk, QAPair

log = get_logger(__name__)


def _render_question_prompt(
    template: str, *, question_type: str, primary: str, neighbors: list[str]
) -> str:
    del question_type  # per-type templates hardcode the type in prose; no placeholder
    neighbor_block = "\n\n---\n\n".join(neighbors) if neighbors else "(none)"
    return template.format(
        primary_chunk=primary,
        neighbor_chunks=neighbor_block,
    )


def _render_answer_prompt(
    template: str,
    *,
    question_type: str,
    question: str,
    primary: str,
    neighbors: list[str],
) -> str:
    neighbor_block = "\n\n---\n\n".join(neighbors) if neighbors else "(none)"
    return template.format(
        question_type=question_type,
        question=question,
        primary_chunk=primary,
        neighbor_chunks=neighbor_block,
    )


def _neighbors(chunks: list[Chunk], idx: int, window: int) -> list[str]:
    if window <= 0:
        return []
    lo = max(0, idx - window)
    hi = min(len(chunks), idx + window + 1)
    return [chunks[i]["text"] for i in range(lo, hi) if i != idx]


def _targets_from_distribution(n: int, dist: dict[str, float]) -> dict[str, int]:
    # Largest-remainder method — deterministic integer counts summing to n.
    raw = {k: n * v for k, v in dist.items()}
    floors = {k: int(v) for k, v in raw.items()}
    remaining = n - sum(floors.values())
    order = sorted(dist.keys(), key=lambda k: raw[k] - floors[k], reverse=True)
    for k in order[:remaining]:
        floors[k] += 1
    return floors


def _call_openai(
    prompt: str, model: str, cfg: QAGenerationConfig, temperature: float
) -> str:
    from openai import AuthenticationError, OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    client = OpenAI(api_key=api_key, timeout=cfg.request_timeout_s)
    last_err: Exception | None = None
    for attempt in range(cfg.max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                max_completion_tokens=cfg.max_tokens,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content or ""
        except AuthenticationError:
            # Auth errors don't get better by retrying — fail fast.
            raise
        except Exception as e:  # noqa: BLE001
            last_err = e
            backoff = 2**attempt
            log.warning(
                "OpenAI call failed (attempt %d/%d): %s — retrying in %ds",
                attempt + 1,
                cfg.max_retries,
                e,
                backoff,
            )
            time.sleep(backoff)
    raise RuntimeError(f"OpenAI call failed after {cfg.max_retries} attempts: {last_err}")


def _question_model(gen_cfg: QAGenerationConfig, qtype: str) -> str:
    if qtype == "factoid":
        return gen_cfg.models.factoid_question
    if qtype == "multihop":
        return gen_cfg.models.multihop_question
    if qtype == "synthesis":
        return gen_cfg.models.synthesis_question
    raise ValueError(f"Unknown question type: {qtype!r}")


def _parse_response(raw: str) -> dict | None:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _build_work_list(
    targets: dict[str, int],
    available_indices: list[int],
    rng: random.Random,
) -> list[tuple[str, int]]:
    """Build a round-robin interleaved work list of (type, chunk_idx) pairs.

    Each type draws from its own shuffled view of chunk indices so the same
    chunk isn't consistently paired with the same type across the batch, and
    types interleave in the output so a reviewer sees a mix rather than a
    block of factoids followed by a block of multihops.
    """
    if not available_indices:
        return []
    n_avail = len(available_indices)
    per_type: dict[str, list[tuple[str, int]]] = {}
    for qtype, count in targets.items():
        shuffled = list(available_indices)
        rng.shuffle(shuffled)
        per_type[qtype] = [(qtype, shuffled[i % n_avail]) for i in range(count)]

    iters = {qtype: iter(items) for qtype, items in per_type.items()}
    work: list[tuple[str, int]] = []
    while True:
        progressed = False
        for qtype in targets:
            item = next(iters[qtype], None)
            if item is not None:
                work.append(item)
                progressed = True
        if not progressed:
            break
    return work


def generate_qa(
    chunks_path: str | Path,
    qa_cfg: QAConfig,
    question_prompt_paths: dict[str, str | Path],
    answer_prompt_path: str | Path,
    limit: int | None = None,
    seed: int | None = None,
    qa_id_offset: int = 0,
    exclude_chunk_ids: set[str] | None = None,
    type_override: str | None = None,
) -> list[QAPair]:
    """Generate a stratified first-pass batch of synthetic QA pairs from chunks.

    Each pair is produced in two LLM calls:
      1. Question generation (temperature = qa_cfg.generation.question_temperature)
      2. Answer generation for the produced question (temperature = qa_cfg.generation.answer_temperature)

    Produces up to ``limit`` (or ``qa_cfg.initial_batch_size``) candidates,
    stratified by ``qa_cfg.type_distribution``. All pairs are returned with
    ``validated=False`` for downstream human review.

    Pass ``seed`` for reproducible stratification and chunk selection.
    Pass ``qa_id_offset`` to start qa_id numbering from a given index — used
    when resuming generation so new ids don't collide with previously decided
    ones. Pass ``exclude_chunk_ids`` to skip chunks already mined for QA.
    Pass ``type_override`` to restrict generation to a single type (used by
    the per-type top-up loop in filter_qa.py).
    """
    gen_cfg = qa_cfg.generation
    if gen_cfg.provider != "openai":
        raise ValueError(
            f"Unsupported QA generation provider: {gen_cfg.provider!r}. "
            "Only 'openai' is currently implemented."
        )

    n = limit if limit is not None else qa_cfg.initial_batch_size
    chunks = cast(list[Chunk], read_jsonl(chunks_path))
    if not chunks:
        raise ValueError(f"No chunks found at {chunks_path}")

    if type_override is not None:
        if type_override not in qa_cfg.type_distribution:
            raise ValueError(
                f"type_override={type_override!r} is not in type_distribution "
                f"keys {sorted(qa_cfg.type_distribution)}."
            )
        distribution = {type_override: 1.0}
    else:
        distribution = qa_cfg.type_distribution

    missing_types = set(distribution) - set(question_prompt_paths)
    if missing_types:
        raise ValueError(
            f"question_prompt_paths is missing entries for types: {sorted(missing_types)}. "
            f"Got keys: {sorted(question_prompt_paths)}."
        )
    q_templates = {qtype: read_text(path) for qtype, path in question_prompt_paths.items()}
    a_template = read_text(answer_prompt_path)
    targets = _targets_from_distribution(n, distribution)
    log.info("QA generation targets for n=%d: %s", n, targets)

    exclude = exclude_chunk_ids or set()
    available_indices = [i for i, c in enumerate(chunks) if c["chunk_id"] not in exclude]
    if not available_indices:
        raise ValueError(
            f"No chunks available after excluding {len(exclude)} ids from {len(chunks)} total."
        )
    if exclude:
        log.info(
            "Excluding %d chunks from sampling pool (%d available, %d total)",
            len(chunks) - len(available_indices),
            len(available_indices),
            len(chunks),
        )

    rng = random.Random(seed)
    work = _build_work_list(targets, available_indices, rng)

    pairs: list[QAPair] = []
    skipped_q = skipped_a = parse_err = missing_field = 0
    for i, (qtype, chunk_idx) in enumerate(work):
        primary = chunks[chunk_idx]
        neighbors = (
            _neighbors(chunks, chunk_idx, qa_cfg.neighbor_window)
            if qtype != "factoid"
            else []
        )

        q_prompt = _render_question_prompt(
            q_templates[qtype],
            question_type=qtype,
            primary=primary["text"],
            neighbors=neighbors,
        )
        q_model = _question_model(gen_cfg, qtype)
        q_raw = _call_openai(q_prompt, q_model, gen_cfg, gen_cfg.question_temperature)
        q_parsed = _parse_response(q_raw)
        if q_parsed is None:
            parse_err += 1
            log.warning(
                "Could not parse question response for chunk %s (type=%s)",
                primary["chunk_id"],
                qtype,
            )
            continue
        if q_parsed.get("skip"):
            skipped_q += 1
            log.info(
                "Skipped chunk %s at question step (type=%s): %s",
                primary["chunk_id"],
                qtype,
                q_parsed.get("reason"),
            )
            continue
        question = q_parsed.get("question")
        if not question:
            missing_field += 1
            log.warning(
                "Question response missing 'question' field for chunk %s (type=%s): %r",
                primary["chunk_id"],
                qtype,
                q_raw,
            )
            continue
        question = str(question).strip()

        a_prompt = _render_answer_prompt(
            a_template,
            question_type=qtype,
            question=question,
            primary=primary["text"],
            neighbors=neighbors,
        )
        a_raw = _call_openai(a_prompt, gen_cfg.models.answer, gen_cfg, gen_cfg.answer_temperature)
        a_parsed = _parse_response(a_raw)
        if a_parsed is None:
            parse_err += 1
            log.warning(
                "Could not parse answer response for chunk %s (type=%s, question=%r)",
                primary["chunk_id"],
                qtype,
                question,
            )
            continue
        if a_parsed.get("skip"):
            skipped_a += 1
            log.info(
                "Skipped at answer step (chunk=%s, type=%s, question=%r): %s",
                primary["chunk_id"],
                qtype,
                question,
                a_parsed.get("reason"),
            )
            continue
        answer = a_parsed.get("answer")
        if not answer:
            missing_field += 1
            log.warning(
                "Answer response missing 'answer' field for chunk %s (type=%s): %r",
                primary["chunk_id"],
                qtype,
                a_raw,
            )
            continue
        answer = str(answer).strip()

        pairs.append(
            QAPair(
                qa_id=f"qa_{qa_id_offset + i:04d}",
                question=question,
                answer=answer,
                source_chunk_id=primary["chunk_id"],
                type=qtype,
                validated=False,
            )
        )
        log.info(
            "Generated %d/%d (type=%s, chunk=%s)",
            len(pairs),
            n,
            qtype,
            primary["chunk_id"],
        )

    type_counts = {t: 0 for t in targets}
    for p in pairs:
        type_counts[p["type"]] += 1
    log.info(
        "QA generation done: produced=%d/%d  by_type=%s  "
        "skipped_q=%d skipped_a=%d parse_err=%d missing_field=%d",
        len(pairs),
        n,
        type_counts,
        skipped_q,
        skipped_a,
        parse_err,
        missing_field,
    )

    return pairs
