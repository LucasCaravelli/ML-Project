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
    neighbor_block = "\n\n---\n\n".join(neighbors) if neighbors else "(none)"
    return template.format(
        question_type=question_type,
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


def _call_openai(prompt: str, cfg: QAGenerationConfig, temperature: float) -> str:
    from openai import OpenAI

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
                model=cfg.model_name,
                temperature=temperature,
                max_tokens=cfg.max_tokens,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content or ""
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


def _parse_response(raw: str) -> dict | None:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def generate_qa(
    chunks_path: str | Path,
    qa_cfg: QAConfig,
    question_prompt_path: str | Path,
    answer_prompt_path: str | Path,
    limit: int | None = None,
) -> list[QAPair]:
    """Generate a stratified first-pass batch of synthetic QA pairs from chunks.

    Each pair is produced in two LLM calls:
      1. Question generation (temperature = qa_cfg.generation.question_temperature)
      2. Answer generation for the produced question (temperature = qa_cfg.generation.answer_temperature)

    Produces up to ``limit`` (or ``qa_cfg.initial_batch_size``) candidates,
    stratified by ``qa_cfg.type_distribution``. All pairs are returned with
    ``validated=False`` for downstream human review.
    """
    n = limit if limit is not None else qa_cfg.initial_batch_size
    chunks = cast(list[Chunk], read_jsonl(chunks_path))
    if not chunks:
        raise ValueError(f"No chunks found at {chunks_path}")

    q_template = read_text(question_prompt_path)
    a_template = read_text(answer_prompt_path)
    targets = _targets_from_distribution(n, qa_cfg.type_distribution)
    log.info("QA generation targets for n=%d: %s", n, targets)

    indices = list(range(len(chunks)))
    random.shuffle(indices)

    work: list[tuple[str, int]] = []
    cursor = 0
    for qtype, count in targets.items():
        for _ in range(count):
            work.append((qtype, indices[cursor % len(indices)]))
            cursor += 1

    gen_cfg = qa_cfg.generation
    pairs: list[QAPair] = []
    for i, (qtype, chunk_idx) in enumerate(work):
        primary = chunks[chunk_idx]
        neighbors = (
            _neighbors(chunks, chunk_idx, qa_cfg.neighbor_window)
            if qtype != "factoid"
            else []
        )

        q_prompt = _render_question_prompt(
            q_template,
            question_type=qtype,
            primary=primary["text"],
            neighbors=neighbors,
        )
        q_raw = _call_openai(q_prompt, gen_cfg, gen_cfg.question_temperature)
        q_parsed = _parse_response(q_raw)
        if q_parsed is None:
            log.warning(
                "Could not parse question response for chunk %s (type=%s)",
                primary["chunk_id"],
                qtype,
            )
            continue
        if q_parsed.get("skip"):
            log.info(
                "Skipped chunk %s at question step (type=%s): %s",
                primary["chunk_id"],
                qtype,
                q_parsed.get("reason"),
            )
            continue
        question = q_parsed.get("question")
        if not question:
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
        a_raw = _call_openai(a_prompt, gen_cfg, gen_cfg.answer_temperature)
        a_parsed = _parse_response(a_raw)
        if a_parsed is None:
            log.warning(
                "Could not parse answer response for chunk %s (type=%s, question=%r)",
                primary["chunk_id"],
                qtype,
                question,
            )
            continue
        if a_parsed.get("skip"):
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
                qa_id=f"qa_{i:04d}",
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

    return pairs
