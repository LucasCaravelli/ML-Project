from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_cr.config import QAConfig, QAGenerationConfig
from rag_cr.io import read_jsonl, write_jsonl
from rag_cr.qa_gen import _targets_from_distribution, generate_qa


def test_stratification_sums_to_n() -> None:
    dist = {"factoid": 0.40, "multihop": 0.35, "synthesis": 0.25}
    for n in (0, 1, 7, 10, 120, 1000):
        targets = _targets_from_distribution(n, dist)
        assert sum(targets.values()) == n
        assert set(targets.keys()) == set(dist.keys())
        assert all(count >= 0 for count in targets.values())


def test_stratification_matches_distribution_at_scale() -> None:
    dist = {"factoid": 0.40, "multihop": 0.35, "synthesis": 0.25}
    n = 1000
    targets = _targets_from_distribution(n, dist)
    assert targets == {"factoid": 400, "multihop": 350, "synthesis": 250}


def test_stratification_deterministic() -> None:
    dist = {"factoid": 0.40, "multihop": 0.35, "synthesis": 0.25}
    a = _targets_from_distribution(13, dist)
    b = _targets_from_distribution(13, dist)
    assert a == b


def test_qapair_jsonl_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"
    pairs = [
        {
            "qa_id": "qa_0000",
            "question": "What is alpha?",
            "answer": "first letter",
            "source_chunk_id": "c_0",
            "type": "factoid",
            "validated": False,
        },
        {
            "qa_id": "qa_0001",
            "question": "Which chunks combine to define beta?",
            "answer": "primary and neighbor",
            "source_chunk_id": "c_1",
            "type": "multihop",
            "validated": True,
        },
    ]
    write_jsonl(path, pairs)
    loaded = read_jsonl(path)
    assert loaded == pairs


def test_prompt_files_present_and_have_placeholders(project_root: Path) -> None:
    q_prompt = (project_root / "prompts" / "qa_generation.txt").read_text(encoding="utf-8")
    a_prompt = (project_root / "prompts" / "qa_answer.txt").read_text(encoding="utf-8")

    for placeholder in ("{question_type}", "{primary_chunk}", "{neighbor_chunks}"):
        assert placeholder in q_prompt, f"{placeholder} missing from qa_generation.txt"
        assert placeholder in a_prompt, f"{placeholder} missing from qa_answer.txt"
    assert "{question}" in a_prompt, "{question} missing from qa_answer.txt"


def _fake_qa_cfg() -> QAConfig:
    return QAConfig(
        target_count=6,
        initial_batch_size=6,
        source_chunk_size=512,
        neighbor_window=1,
        type_distribution={"factoid": 0.5, "multihop": 0.25, "synthesis": 0.25},
        generation=QAGenerationConfig(
            provider="openai",
            model_name="gpt-test",
            question_temperature=0.4,
            answer_temperature=0.1,
            max_tokens=256,
            request_timeout_s=10,
            max_retries=1,
        ),
    )


ANSWER_PROMPT_MARKER = "producing the ground-truth ANSWER"


def _is_answer_prompt(prompt: str) -> bool:
    return ANSWER_PROMPT_MARKER in prompt


def _write_fake_chunks(path: Path, n: int) -> None:
    write_jsonl(
        path,
        [
            {
                "chunk_id": f"c_{i}",
                "size": 512,
                "start_char": i * 100,
                "end_char": (i + 1) * 100,
                "text": f"chunk text number {i} with placeholder content.",
            }
            for i in range(n)
        ],
    )


def test_generate_qa_pipeline_shape(
    tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    _write_fake_chunks(chunks_path, 4)

    def fake_call(prompt: str, cfg: QAGenerationConfig, temperature: float) -> str:
        if _is_answer_prompt(prompt):
            return json.dumps({"answer": "placeholder content"})
        return json.dumps({"question": "What is the chunk text?"})

    monkeypatch.setattr("rag_cr.qa_gen._call_openai", fake_call)

    qa_cfg = _fake_qa_cfg()
    pairs = generate_qa(
        chunks_path=chunks_path,
        qa_cfg=qa_cfg,
        question_prompt_path=project_root / "prompts" / "qa_generation.txt",
        answer_prompt_path=project_root / "prompts" / "qa_answer.txt",
        limit=qa_cfg.initial_batch_size,
        seed=0,
    )

    assert len(pairs) == qa_cfg.initial_batch_size
    required_fields = {"qa_id", "question", "answer", "source_chunk_id", "type", "validated"}
    for pair in pairs:
        assert required_fields.issubset(pair.keys())
        assert pair["validated"] is False
        assert pair["type"] in qa_cfg.type_distribution
        assert pair["source_chunk_id"].startswith("c_")
        assert pair["question"]
        assert pair["answer"]

    type_counts = {t: 0 for t in qa_cfg.type_distribution}
    for pair in pairs:
        type_counts[pair["type"]] += 1
    expected = _targets_from_distribution(qa_cfg.initial_batch_size, qa_cfg.type_distribution)
    assert type_counts == expected


def test_generate_qa_skips_unparseable_response(
    tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    _write_fake_chunks(chunks_path, 1)

    def fake_call(prompt: str, cfg: QAGenerationConfig, temperature: float) -> str:
        return "not valid json"

    monkeypatch.setattr("rag_cr.qa_gen._call_openai", fake_call)

    qa_cfg = _fake_qa_cfg()
    pairs = generate_qa(
        chunks_path=chunks_path,
        qa_cfg=qa_cfg,
        question_prompt_path=project_root / "prompts" / "qa_generation.txt",
        answer_prompt_path=project_root / "prompts" / "qa_answer.txt",
        limit=3,
        seed=0,
    )
    assert pairs == []


def test_generate_qa_skips_when_question_step_skips(
    tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    _write_fake_chunks(chunks_path, 2)

    def fake_call(prompt: str, cfg: QAGenerationConfig, temperature: float) -> str:
        if _is_answer_prompt(prompt):
            pytest.fail("answer prompt should not be reached when question step skips")
        return json.dumps({"skip": True, "reason": "too thin"})

    monkeypatch.setattr("rag_cr.qa_gen._call_openai", fake_call)

    qa_cfg = _fake_qa_cfg()
    pairs = generate_qa(
        chunks_path=chunks_path,
        qa_cfg=qa_cfg,
        question_prompt_path=project_root / "prompts" / "qa_generation.txt",
        answer_prompt_path=project_root / "prompts" / "qa_answer.txt",
        limit=4,
        seed=0,
    )
    assert pairs == []


def test_generate_qa_skips_when_answer_step_skips(
    tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    _write_fake_chunks(chunks_path, 2)

    def fake_call(prompt: str, cfg: QAGenerationConfig, temperature: float) -> str:
        if _is_answer_prompt(prompt):
            return json.dumps({"skip": True, "reason": "cannot fit word cap"})
        return json.dumps({"question": "What is the chunk text?"})

    monkeypatch.setattr("rag_cr.qa_gen._call_openai", fake_call)

    qa_cfg = _fake_qa_cfg()
    pairs = generate_qa(
        chunks_path=chunks_path,
        qa_cfg=qa_cfg,
        question_prompt_path=project_root / "prompts" / "qa_generation.txt",
        answer_prompt_path=project_root / "prompts" / "qa_answer.txt",
        limit=4,
        seed=0,
    )
    assert pairs == []


def test_generate_qa_rejects_unsupported_provider(
    tmp_path: Path, project_root: Path
) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    _write_fake_chunks(chunks_path, 1)

    qa_cfg = _fake_qa_cfg()
    bad_cfg = QAConfig(
        target_count=qa_cfg.target_count,
        initial_batch_size=qa_cfg.initial_batch_size,
        source_chunk_size=qa_cfg.source_chunk_size,
        neighbor_window=qa_cfg.neighbor_window,
        type_distribution=qa_cfg.type_distribution,
        generation=QAGenerationConfig(
            provider="ollama",
            model_name=qa_cfg.generation.model_name,
            question_temperature=qa_cfg.generation.question_temperature,
            answer_temperature=qa_cfg.generation.answer_temperature,
            max_tokens=qa_cfg.generation.max_tokens,
            request_timeout_s=qa_cfg.generation.request_timeout_s,
            max_retries=qa_cfg.generation.max_retries,
        ),
    )

    with pytest.raises(ValueError, match="Unsupported QA generation provider"):
        generate_qa(
            chunks_path=chunks_path,
            qa_cfg=bad_cfg,
            question_prompt_path=project_root / "prompts" / "qa_generation.txt",
            answer_prompt_path=project_root / "prompts" / "qa_answer.txt",
            limit=2,
            seed=0,
        )
