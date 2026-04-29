from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_cr.config import QAConfig, QAGenerationConfig, QAModelsConfig
from rag_cr.qa_filter import compute_deficits, filter_qa
from rag_cr.types import Chunk, QAPair


def _qa_cfg() -> QAConfig:
    return QAConfig(
        target_count=12,
        initial_batch_size=12,
        source_chunk_size=512,
        neighbor_window=1,
        type_distribution={"factoid": 1 / 3, "multihop": 1 / 3, "synthesis": 1 / 3},
        generation=QAGenerationConfig(
            provider="openai",
            models=QAModelsConfig(
                factoid_question="m-fq",
                multihop_question="m-mq",
                synthesis_question="m-sq",
                answer="m-a",
                judge="m-j",
            ),
            question_temperature=0.4,
            answer_temperature=0.1,
            judge_temperature=0.0,
            max_tokens=256,
            request_timeout_s=10,
            max_retries=1,
            primary_only_f1_threshold=0.8,
            max_topup_rounds=3,
        ),
    )


def _chunks() -> list[Chunk]:
    return [
        {
            "chunk_id": f"c_{i}",
            "size": 512,
            "start_char": i * 100,
            "end_char": (i + 1) * 100,
            "text": f"chunk {i} body text",
        }
        for i in range(4)
    ]


def _pair(qa_id: str, qtype: str, *, answer: str = "gold answer phrase") -> QAPair:
    return {
        "qa_id": qa_id,
        "question": f"q for {qa_id}?",
        "answer": answer,
        "source_chunk_id": "c_1",
        "type": qtype,
        "validated": False,
    }


ANSWER_PROMPT_MARKER = "producing the ground-truth ANSWER"
JUDGE_PROMPT_MARKER = "evaluator for a retrieval-augmented generation benchmark"


def _is_answer_prompt(prompt: str) -> bool:
    return ANSWER_PROMPT_MARKER in prompt


def _is_judge_prompt(prompt: str) -> bool:
    return JUDGE_PROMPT_MARKER in prompt


def _load_prompts(project_root: Path) -> tuple[str, str]:
    answer = (project_root / "prompts" / "qa_answer.txt").read_text(encoding="utf-8")
    judge = (project_root / "prompts" / "judge.txt").read_text(encoding="utf-8")
    return answer, judge


def test_factoid_skips_primary_only_check_and_uses_judge(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer_prompt, judge_prompt = _load_prompts(project_root)
    calls: list[str] = []

    def fake_call(prompt: str, model: str, cfg, temperature: float) -> str:
        if _is_judge_prompt(prompt):
            calls.append("judge")
            return json.dumps({"keep": True, "reason": "ok"})
        if _is_answer_prompt(prompt):
            calls.append("answer")
            return json.dumps({"answer": "anything"})
        raise AssertionError("unexpected prompt kind")

    monkeypatch.setattr("rag_cr.qa_filter._call_openai", fake_call)

    pairs = [_pair("qa_0001", "factoid")]
    kept, rejected = filter_qa(
        pairs, _chunks(), _qa_cfg(), answer_prompt=answer_prompt, judge_prompt=judge_prompt
    )

    assert len(kept) == 1
    assert kept[0]["validated"] is True
    assert rejected == []
    # factoid should have called the judge once and never the primary-only answer step
    assert calls == ["judge"]


def test_primary_only_match_drops_multihop(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer_prompt, judge_prompt = _load_prompts(project_root)
    judge_invoked = []

    def fake_call(prompt: str, model: str, cfg, temperature: float) -> str:
        if _is_judge_prompt(prompt):
            judge_invoked.append(True)
            return json.dumps({"keep": True, "reason": "should not be reached"})
        if _is_answer_prompt(prompt):
            return json.dumps({"answer": "gold answer phrase"})
        raise AssertionError("unexpected prompt kind")

    monkeypatch.setattr("rag_cr.qa_filter._call_openai", fake_call)

    pairs = [_pair("qa_0001", "multihop", answer="gold answer phrase")]
    kept, rejected = filter_qa(
        pairs, _chunks(), _qa_cfg(), answer_prompt=answer_prompt, judge_prompt=judge_prompt
    )

    assert kept == []
    assert len(rejected) == 1
    rec = rejected[0]
    assert rec["qa_id"] == "qa_0001"
    assert rec["type"] == "multihop"
    assert rec["filter"] == "primary_only"
    assert "F1=" in rec["reason"]
    assert judge_invoked == []  # judge must NOT be called when primary_only fires


def test_primary_only_mismatch_passes_to_judge(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer_prompt, judge_prompt = _load_prompts(project_root)

    def fake_call(prompt: str, model: str, cfg, temperature: float) -> str:
        if _is_judge_prompt(prompt):
            return json.dumps({"keep": True, "reason": "supported"})
        if _is_answer_prompt(prompt):
            # Completely different from gold → low F1
            return json.dumps({"answer": "totally unrelated content"})
        raise AssertionError("unexpected prompt kind")

    monkeypatch.setattr("rag_cr.qa_filter._call_openai", fake_call)

    pairs = [_pair("qa_0002", "synthesis", answer="gold answer phrase here")]
    kept, rejected = filter_qa(
        pairs, _chunks(), _qa_cfg(), answer_prompt=answer_prompt, judge_prompt=judge_prompt
    )

    assert len(kept) == 1
    assert kept[0]["validated"] is True
    assert rejected == []


def test_judge_drop_branch_records_reason(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer_prompt, judge_prompt = _load_prompts(project_root)

    def fake_call(prompt: str, model: str, cfg, temperature: float) -> str:
        if _is_judge_prompt(prompt):
            return json.dumps({"keep": False, "reason": "wrong type"})
        if _is_answer_prompt(prompt):
            return json.dumps({"answer": "different content"})
        raise AssertionError("unexpected prompt kind")

    monkeypatch.setattr("rag_cr.qa_filter._call_openai", fake_call)

    pairs = [_pair("qa_0003", "factoid", answer="gold")]
    kept, rejected = filter_qa(
        pairs, _chunks(), _qa_cfg(), answer_prompt=answer_prompt, judge_prompt=judge_prompt
    )

    assert kept == []
    assert len(rejected) == 1
    rec = rejected[0]
    assert rec["filter"] == "judge"
    assert rec["reason"] == "wrong type"
    assert set(rec.keys()) == {"qa_id", "type", "filter", "reason"}


def test_judge_unparseable_response_drops_with_reason(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer_prompt, judge_prompt = _load_prompts(project_root)

    def fake_call(prompt: str, model: str, cfg, temperature: float) -> str:
        if _is_judge_prompt(prompt):
            return "not json"
        return json.dumps({"answer": "different"})

    monkeypatch.setattr("rag_cr.qa_filter._call_openai", fake_call)

    pairs = [_pair("qa_0004", "factoid", answer="gold")]
    kept, rejected = filter_qa(
        pairs, _chunks(), _qa_cfg(), answer_prompt=answer_prompt, judge_prompt=judge_prompt
    )

    assert kept == []
    assert rejected[0]["filter"] == "judge"
    assert "unparseable" in rejected[0]["reason"]


def test_filter_routes_models_correctly(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Primary-only check uses the answer model; judge call uses the judge model."""
    answer_prompt, judge_prompt = _load_prompts(project_root)
    seen: list[tuple[str, str]] = []

    def fake_call(prompt: str, model: str, cfg, temperature: float) -> str:
        if _is_judge_prompt(prompt):
            seen.append(("judge", model))
            return json.dumps({"keep": True, "reason": "ok"})
        if _is_answer_prompt(prompt):
            seen.append(("answer", model))
            return json.dumps({"answer": "completely unrelated"})
        raise AssertionError("unexpected prompt kind")

    monkeypatch.setattr("rag_cr.qa_filter._call_openai", fake_call)

    pairs = [_pair("qa_0005", "multihop", answer="gold phrase")]
    cfg = _qa_cfg()
    filter_qa(pairs, _chunks(), cfg, answer_prompt=answer_prompt, judge_prompt=judge_prompt)

    answer_models = {m for k, m in seen if k == "answer"}
    judge_models = {m for k, m in seen if k == "judge"}
    assert answer_models == {cfg.generation.models.answer}
    assert judge_models == {cfg.generation.models.judge}


def test_compute_deficits_counts_validated_and_raw() -> None:
    quota = {"factoid": 134, "multihop": 133, "synthesis": 133}
    validated = {"factoid": 80, "multihop": 50, "synthesis": 0}
    raw_unfiltered = {"factoid": 40, "multihop": 0, "synthesis": 10}
    deficits = compute_deficits(quota, validated, raw_unfiltered)
    assert deficits == {"factoid": 14, "multihop": 83, "synthesis": 123}


def test_compute_deficits_clamps_negative_to_zero() -> None:
    quota = {"factoid": 100}
    validated = {"factoid": 90}
    raw_unfiltered = {"factoid": 30}
    assert compute_deficits(quota, validated, raw_unfiltered) == {"factoid": 0}


def test_rejected_records_have_required_schema_fields(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer_prompt, judge_prompt = _load_prompts(project_root)

    def fake_call(prompt: str, model: str, cfg, temperature: float) -> str:
        if _is_judge_prompt(prompt):
            return json.dumps({"keep": False, "reason": "bad"})
        if _is_answer_prompt(prompt):
            return json.dumps({"answer": "matches gold answer phrase"})
        raise AssertionError

    monkeypatch.setattr("rag_cr.qa_filter._call_openai", fake_call)

    pairs = [
        _pair("qa_0010", "factoid", answer="gold"),
        _pair("qa_0011", "multihop", answer="gold answer phrase"),
        _pair("qa_0012", "synthesis", answer="gold answer phrase"),
    ]
    _, rejected = filter_qa(
        pairs, _chunks(), _qa_cfg(), answer_prompt=answer_prompt, judge_prompt=judge_prompt
    )
    required = {"qa_id", "type", "filter", "reason"}
    for rec in rejected:
        assert required.issubset(rec.keys())
        assert rec["filter"] in {"primary_only", "judge"}
