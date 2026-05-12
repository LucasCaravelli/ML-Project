"""End-to-end retrieve→generate→score orchestration for a configured system."""

from __future__ import annotations

from pathlib import Path

from .config import Config, load_config
from .io import create_run_dir, get_qa_validated_path, read_jsonl, write_json, write_jsonl
from .metrics import score
from .systems import build_system
from .types import QAPair, ScoreDict


def _load_qa(qa_path: str | Path | None, config: Config) -> list[QAPair]:
    p = Path(qa_path) if qa_path is not None else get_qa_validated_path(config.paths.artifacts_dir)
    if not p.exists():
        raise FileNotFoundError(
            f"QA file not found at {p}. "
            "Run experiments/validate_qa.py first, or pass an explicit qa_path."
        )
    return [QAPair(**row) for row in read_jsonl(p)]  # type: ignore[misc]


def run(
    system_name: str,
    qa_path: str | Path | None = None,
    config: Config | None = None,
    limit: int | None = None,
) -> Path:
    """Run one evaluation system over a QA set and return the results directory."""
    if config is None:
        config = load_config()

    system = build_system(system_name, config)
    qa_pairs = _load_qa(qa_path, config)
    if limit is not None:
        qa_pairs = qa_pairs[:limit]

    run_dir = create_run_dir(config.paths.results_dir, system_name)
    predictions: list[dict] = []
    all_scores: list[ScoreDict] = []

    for qa in qa_pairs:
        prediction, passages = system.answer(qa["question"], qa_id=qa["qa_id"])
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
