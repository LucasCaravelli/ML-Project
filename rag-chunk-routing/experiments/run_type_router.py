"""Type-aware chunk-size routing sanity baseline (no GPU / LLM needed).

Policy: for each test question, assign the chunk size that achieves the
highest *mean* F1 for that question's type, computed from the pre-built
eval grid. F1 for each (question, size) pair is then looked up from the
same eval grid — no new inference is required.

This answers: does knowing only the question type suffice to recover
oracle headroom, without any training?

Run from rag-chunk-routing/:
    python experiments/run_type_router.py --config configs/base.yaml
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from rag_cr import Config, load_config
from rag_cr.io import create_run_dir, read_jsonl, write_json, write_jsonl


def _read_oracle_gap(artifacts_dir: Path) -> tuple[float | None, float | None]:
    path = artifacts_dir / "baselines" / "oracle_gap.json"
    if not path.exists():
        return None, None
    data = json.loads(path.read_text())
    return data.get("oracle_f1_mean"), data.get("best_baseline_f1")


def _gap_closure(
    router_f1: float, baseline_f1: float | None, oracle_f1: float | None
) -> float | None:
    if baseline_f1 is None or oracle_f1 is None:
        return None
    denom = oracle_f1 - baseline_f1
    if abs(denom) < 1e-9:
        return None
    return (router_f1 - baseline_f1) / denom


def main(config: Config, config_path: str) -> None:
    artifacts_dir = config.paths.artifacts_dir
    results_dir = config.paths.results_dir
    eval_grid_path = artifacts_dir / "oracle" / "eval_grid.jsonl"
    chunk_sizes: list[int] = list(config.chunking.sizes)

    if not eval_grid_path.exists():
        raise FileNotFoundError(
            f"Eval grid not found at {eval_grid_path}. Run compute_oracle first."
        )

    # Load eval grid for test split: qa_id -> {size -> f1}
    f1_grid: dict[str, dict[int, float]] = defaultdict(dict)
    type_by_id: dict[str, str] = {}
    for row in read_jsonl(eval_grid_path):
        if row.get("split") != "test":
            continue
        size = int(row["chunk_size"])
        if size not in chunk_sizes:
            continue
        qa_id = row["qa_id"]
        f1_grid[qa_id][size] = float(row["f1"])
        type_by_id[qa_id] = row.get("type", "factoid")

    if not f1_grid:
        raise RuntimeError("No test rows found in eval grid for the active chunk sizes.")

    # Compute per-type mean F1 by size, then pick the best size for each type
    type_size_f1s: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for qa_id, size_f1 in f1_grid.items():
        qtype = type_by_id[qa_id]
        for size, f1 in size_f1.items():
            type_size_f1s[qtype][size].append(f1)

    type_best_size: dict[str, int] = {}
    print("Per-type mean F1 from eval grid (test split):")
    for qtype in sorted(type_size_f1s):
        mean_by_size = {s: float(np.mean(fs)) for s, fs in type_size_f1s[qtype].items()}
        best_size = max(mean_by_size, key=mean_by_size.__getitem__)
        type_best_size[qtype] = best_size
        size_str = "  ".join(f"size-{s}: {v:.4f}" for s, v in sorted(mean_by_size.items()))
        print(f"  {qtype}: best={best_size}  {size_str}")
    print()

    # Apply policy: each question gets its type's best size; look up F1 from eval grid
    pred_rows = []
    f1_scores: list[float] = []
    for qa_id in sorted(f1_grid):
        qtype = type_by_id[qa_id]
        pred_size = type_best_size.get(qtype, min(chunk_sizes))
        f1 = f1_grid[qa_id].get(pred_size, 0.0)
        f1_scores.append(f1)
        pred_rows.append({
            "qa_id": qa_id,
            "question_type": qtype,
            "predicted_chunk_size": pred_size,
            "f1": f1,
        })

    mean_f1 = float(np.mean(f1_scores))
    oracle_f1, baseline_f1 = _read_oracle_gap(artifacts_dir)
    gap = _gap_closure(mean_f1, baseline_f1, oracle_f1)

    print(f"Type-aware policy on test ({len(pred_rows)} examples):")
    print(f"  mean F1              : {mean_f1:.4f}")
    if baseline_f1 is not None:
        print(f"  best baseline F1     : {baseline_f1:.4f}")
    if oracle_f1 is not None:
        print(f"  oracle F1            : {oracle_f1:.4f}")
    if gap is not None:
        print(f"  gap_closure_fraction : {gap:.4f}")
    print()

    metrics: dict = {
        "mean_f1": mean_f1,
        "gap_closure_fraction": gap,
        "n_test_examples": len(pred_rows),
        "policy": "type_aware",
        "type_best_sizes": {k: int(v) for k, v in type_best_size.items()},
        "best_baseline_mean_f1_ref": baseline_f1,
        "oracle_mean_f1_ref": oracle_f1,
    }

    run_dir = create_run_dir(results_dir, "type_router")
    write_jsonl(run_dir / "predictions.jsonl", pred_rows)
    write_json(run_dir / "metrics.json", metrics)
    print(f"Predictions → {run_dir / 'predictions.jsonl'}")
    print(f"Metrics     → {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Type-aware routing sanity baseline.")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    main(load_config(args.config), args.config)
