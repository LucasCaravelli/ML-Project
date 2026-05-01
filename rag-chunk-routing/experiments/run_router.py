from __future__ import annotations

import argparse
import json
import logging
import pickle
import subprocess
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score

from rag_cr import Config, load_config, set_seed
from rag_cr.io import create_run_dir, read_jsonl, write_json, write_jsonl
from rag_cr.router.train import load_router_data

log = logging.getLogger(__name__)


def _git_hash() -> str:
    """Return the current git commit hash, or 'unknown' on failure."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _oracle_labels_path(config: Config) -> Path:
    return config.paths.artifacts_dir / "oracle" / "labels.jsonl"


def _load_best_results(results_dir: Path, system_name: str) -> float | None:
    """Find the most recent metrics.json for a given system and return mean_f1, or None."""
    runs = sorted((results_dir / "runs").glob(f"*_{system_name}"), reverse=True)
    for run in runs:
        metrics_path = run / "metrics.json"
        if metrics_path.exists():
            data = json.loads(metrics_path.read_text())
            val = data.get("mean_f1")
            if val is not None:
                return float(val)
    return None


def _gap_closure(router_f1: float, best_baseline_f1: float | None, oracle_f1: float | None) -> float | None:
    """Compute (router - baseline) / (oracle - baseline), or None if reference data missing."""
    if best_baseline_f1 is None or oracle_f1 is None:
        return None
    denom = oracle_f1 - best_baseline_f1
    if abs(denom) < 1e-9:
        return None
    return (router_f1 - best_baseline_f1) / denom


def _best_baseline_f1(results_dir: Path, config: Config) -> float | None:
    """Return the best mean_f1 across all fixed-size baseline runs."""
    best: float | None = None
    for size in config.chunking.sizes:
        f1 = _load_best_results(results_dir, f"fixed_{size}")
        if f1 is not None and (best is None or f1 > best):
            best = f1
    return best


def main(config: Config, config_path: str) -> None:
    """Evaluate the trained router on the test set and write prediction + metric artifacts."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    set_seed(config.project.seed)
    seed = config.project.seed

    print(f"git commit : {_git_hash()}")
    print(f"config     : {config_path}")
    print()

    artifacts_dir = config.paths.artifacts_dir
    results_dir = config.paths.results_dir
    splits_dir = artifacts_dir / "splits"
    oracle_path = _oracle_labels_path(config)
    pkl_path = artifacts_dir / "router" / "best.pkl"
    best_config_path = artifacts_dir / "router" / "best_config.json"

    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Trained router not found at {pkl_path}. "
            "Run experiments/train_router.py first."
        )
    if not oracle_path.exists():
        raise FileNotFoundError(
            f"Oracle labels not found at {oracle_path}. "
            "Run experiments/compute_oracle.py first."
        )

    with pkl_path.open("rb") as f:
        bundle = pickle.load(f)
    extractor = bundle["extractor"]
    classifier = bundle["classifier"]
    best_config_meta: dict[str, Any] = bundle.get("config", {})

    if best_config_path.exists():
        best_config_meta = json.loads(best_config_path.read_text())

    chunk_sizes: list[int] = list(config.chunking.sizes)

    print("Loading test split …")
    test_q, test_t, test_l = load_router_data(splits_dir, oracle_path, "test", chunk_sizes)
    print(f"  test: {len(test_q)} examples")

    # Load raw test rows for qa_id lookup
    test_rows = read_jsonl(splits_dir / "test.jsonl")
    id_to_question = {r["qa_id"]: r for r in test_rows}
    oracle_rows = read_jsonl(oracle_path)
    oracle_by_id: dict[str, int] = {r["qa_id"]: r["best_size"] for r in oracle_rows}

    # Build an ordered list of (qa_id, question, type) that survived the filter
    filtered_ids: list[str] = []
    for row in test_rows:
        best = oracle_by_id.get(row["qa_id"])
        if best is not None and best in chunk_sizes:
            filtered_ids.append(row["qa_id"])

    print("Extracting features and predicting chunk sizes …")
    X_test = extractor.transform(test_q, test_t)
    predictions: np.ndarray = classifier.predict(X_test)

    y_true = np.array(test_l)
    y_pred = predictions

    clf_macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    clf_bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    clf_acc = float(np.mean(y_true == y_pred))

    print(f"  Router classification macro-F1         : {clf_macro_f1:.4f}")
    print(f"  Router classification balanced accuracy : {clf_bal_acc:.4f}")
    print(f"  Router classification accuracy          : {clf_acc:.4f}")
    print()

    # --- RAG pipeline evaluation ---
    print("Running RAG pipeline for each predicted chunk size …")
    pred_rows: list[dict[str, Any]] = []
    rag_scores: list[dict[str, float]] = []
    pipeline_available = True

    try:
        from rag_cr.metrics import score as rag_score
        from rag_cr.systems import FixedSizeSystem

        systems: dict[int, FixedSizeSystem] = {}
        for size in set(int(p) for p in predictions):
            systems[size] = FixedSizeSystem(size, config)

        for qa_id, pred_size in zip(filtered_ids, predictions.tolist()):
            row = id_to_question[qa_id]
            system = systems[int(pred_size)]
            try:
                prediction_text, passages = system.answer(row["question"], qa_id=qa_id)
                s = rag_score(prediction_text, row["answer"], passages)
            except Exception as exc:
                log.warning("Pipeline failed for %s: %s", qa_id, exc)
                prediction_text = ""
                passages = []
                s = {"em": 0.0, "f1": 0.0, "faithfulness": 0.0, "cost_tokens": 0}

            pred_rows.append(
                {
                    "qa_id": qa_id,
                    "question": row["question"],
                    "gold": row["answer"],
                    "predicted_chunk_size": int(pred_size),
                    "oracle_chunk_size": oracle_by_id.get(qa_id),
                    "prediction": prediction_text,
                    **s,
                }
            )
            rag_scores.append({"em": s["em"], "f1": s["f1"], "cost_tokens": s["cost_tokens"]})  # type: ignore[arg-type]

    except Exception as exc:
        log.warning("RAG pipeline unavailable (%s) — writing classification results only.", exc)
        pipeline_available = False
        for qa_id, pred_size in zip(filtered_ids, predictions.tolist()):
            row = id_to_question[qa_id]
            pred_rows.append(
                {
                    "qa_id": qa_id,
                    "question": row["question"],
                    "gold": row["answer"],
                    "predicted_chunk_size": int(pred_size),
                    "oracle_chunk_size": oracle_by_id.get(qa_id),
                }
            )

    # --- Gap closure ---
    mean_f1: float | None = None
    mean_em: float | None = None
    mean_cost: float | None = None

    if pipeline_available and rag_scores:
        mean_f1 = float(np.mean([s["f1"] for s in rag_scores]))
        mean_em = float(np.mean([s["em"] for s in rag_scores]))
        mean_cost = float(np.mean([s["cost_tokens"] for s in rag_scores]))

    baseline_f1 = _best_baseline_f1(results_dir, config)
    oracle_f1 = _load_best_results(results_dir, "oracle")

    if baseline_f1 is None:
        warnings.warn("No baseline results found in results/runs/. gap_closure_fraction set to null.", stacklevel=1)
    if oracle_f1 is None:
        warnings.warn("No oracle results found in results/runs/. gap_closure_fraction set to null.", stacklevel=1)

    gap = _gap_closure(mean_f1 or 0.0, baseline_f1, oracle_f1) if mean_f1 is not None else None

    metrics: dict[str, Any] = {
        "mean_f1": mean_f1,
        "mean_em": mean_em,
        "mean_cost_tokens": mean_cost,
        "macro_f1_router_classification": clf_macro_f1,
        "balanced_accuracy_router_classification": clf_bal_acc,
        "accuracy_router_classification": clf_acc,
        "gap_closure_fraction": gap,
        "n_test_examples": len(filtered_ids),
        "best_baseline_mean_f1_ref": baseline_f1,
        "oracle_mean_f1_ref": oracle_f1,
    }

    # --- Write artifacts ---
    run_dir = create_run_dir(results_dir, "router")

    write_jsonl(run_dir / "predictions.jsonl", pred_rows)
    write_json(run_dir / "metrics.json", metrics)
    write_json(
        run_dir / "meta.json",
        {
            "git_commit": _git_hash(),
            "config_path": config_path,
            "seed": seed,
            "best_config": best_config_meta,
        },
    )

    print(f"Predictions → {run_dir / 'predictions.jsonl'}")
    print(f"Metrics     → {run_dir / 'metrics.json'}")
    print(f"Meta        → {run_dir / 'meta.json'}")
    print()
    print("Key metrics:")
    print(f"  macro_f1_router_classification : {clf_macro_f1:.4f}")
    if gap is not None:
        print(f"  gap_closure_fraction           : {gap:.4f}")
    else:
        print("  gap_closure_fraction           : null (reference runs missing)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the trained chunk-size router on test.")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    main(load_config(args.config), args.config)
