from __future__ import annotations

from pathlib import Path

from .config import Config, load_config
from .io import create_run_dir, read_jsonl, write_json, write_jsonl
from .metrics import score
from .systems import build_system
from .types import QAPair, ScoreDict


_FAKE_QA: list[QAPair] = [
    {
        "qa_id": "fake_001",
        "question": "What is retrieval-augmented generation?",
        "answer": "RAG combines retrieval with generation to produce grounded answers.",
        "source_chunk_id": "chunk_256_0",
        "type": "factoid",
        "validated": True,
    },
    {
        "qa_id": "fake_002",
        "question": "How does chunk size affect retrieval quality?",
        "answer": "Smaller chunks increase precision while larger chunks improve recall.",
        "source_chunk_id": "chunk_512_0",
        "type": "synthesis",
        "validated": True,
    },
    {
        "qa_id": "fake_003",
        "question": "What is reciprocal rank fusion?",
        "answer": "RRF merges ranked lists from multiple retrievers using a harmonic formula.",
        "source_chunk_id": "chunk_128_0",
        "type": "factoid",
        "validated": True,
    },
]


def _load_qa(qa_path: str | Path) -> list[QAPair]:
    p = Path(qa_path)
    if p.exists():
        return [QAPair(**row) for row in read_jsonl(p)]  # type: ignore[misc]
    return list(_FAKE_QA)


def run(system_name: str, qa_path: str | Path, config: Config | None = None) -> Path:
    """Run one evaluation system over a QA set and return the results directory."""
    if config is None:
        config = load_config()

    system = build_system(system_name, config)
    qa_pairs = _load_qa(qa_path)

    run_dir = create_run_dir(config.paths.results_dir, system_name)
    predictions: list[dict] = []
    all_scores: list[ScoreDict] = []

    for qa in qa_pairs:
        prediction, passages = system.answer(qa["question"])
        s = score(prediction, qa["answer"], passages)
        all_scores.append(s)
        predictions.append({
            "qa_id": qa["qa_id"],
            "question": qa["question"],
            "gold": qa["answer"],
            "prediction": prediction,
            **s,
        })

    n = len(all_scores)
    aggregate: dict = {
        "system": system_name,
        "n": n,
        "em":           sum(s["em"] for s in all_scores) / n if n else 0.0,
        "f1":           sum(s["f1"] for s in all_scores) / n if n else 0.0,
        "faithfulness": sum(s["faithfulness"] for s in all_scores) / n if n else 0.0,
        "cost_tokens_total": sum(s["cost_tokens"] for s in all_scores),
    }

    write_jsonl(run_dir / "predictions.jsonl", predictions)
    write_json(run_dir / "metrics.json", aggregate)

    return run_dir
