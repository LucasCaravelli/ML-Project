"""Train the chunk-size router via stratified CV grid search and val re-ranking.

For each {feature_set} × {classifier} combination, runs stratified-K-fold CV
over the train split, ranks the top finalists by val macro-F1, and re-ranks
those finalists by the actual val RAG F1 (looking up answers from the eval
grid) to pick the model that maximizes downstream RAG quality. Saves the
winning extractor + classifier + metadata under artifacts/router/.

Run from rag-chunk-routing/:
    python experiments/train_router.py --config configs/base.yaml
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from rag_cr import Config, load_config, set_seed
from rag_cr.io import read_jsonl
from rag_cr.router.train import (
    load_router_data,
    rank_all_on_val,
    run_cv_grid,
)
from rag_cr.utils import git_hash

_N_FINALISTS = 3


def _oracle_labels_path(config: Config) -> Path:
    return config.paths.artifacts_dir / "oracle" / "labels.jsonl"


def _val_rag_f1(
    extractor,
    classifier,
    val_questions: list[str],
    val_types: list[str],
    val_rows: list[dict],
    config: Config,
) -> float | None:
    """Run RAG on val with predicted sizes; return mean F1, or None if pipeline unavailable."""
    try:
        from rag_cr.metrics import score as rag_score
        from rag_cr.systems import FixedSizeSystem
    except Exception:
        return None

    X_val = extractor.transform(val_questions, val_types)
    predictions = classifier.predict(X_val)

    systems: dict = {}
    for size in set(int(p) for p in predictions):
        systems[size] = FixedSizeSystem(size, config)

    f1_scores: list[float] = []
    for row, pred_size in zip(val_rows, predictions.tolist(), strict=False):
        try:
            pred_text, passages = systems[int(pred_size)].answer(
                row["question"], qa_id=row["qa_id"]
            )
            s = rag_score(pred_text, row["answer"], passages)
            f1_scores.append(float(s["f1"]))
        except Exception:
            f1_scores.append(0.0)

    return sum(f1_scores) / len(f1_scores) if f1_scores else None


def main(config: Config, config_path: str) -> None:
    """Orchestrate router training: CV grid search, val selection, artifact saving."""
    set_seed(config.project.seed)
    seed = config.project.seed

    print(f"git commit : {git_hash()}")
    print(f"config     : {config_path}")
    print()

    artifacts_dir = config.paths.artifacts_dir
    splits_dir = artifacts_dir / "splits"
    oracle_path = _oracle_labels_path(config)
    out_dir = artifacts_dir / "router"
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk_sizes: list[int] = list(config.chunking.sizes)
    print(f"Router chunk classes : {chunk_sizes}")

    if not oracle_path.exists():
        raise FileNotFoundError(
            f"Oracle labels not found at {oracle_path}. "
            "Run experiments/compute_oracle.py first."
        )

    print("Loading train / val splits …")
    train_q, train_t, train_l = load_router_data(splits_dir, oracle_path, "train", chunk_sizes)
    val_q, val_t, val_l = load_router_data(splits_dir, oracle_path, "val", chunk_sizes)
    print(f"  train: {len(train_q)} examples   val: {len(val_q)} examples")
    print()

    # --- Cross-validated grid search on train ---
    print(f"Running {len(config.router.feature_sets)}×{len(config.router.model_names)} grid "
          f"with {config.router.cv_folds}-fold CV …")
    cv_df = run_cv_grid(train_q, train_t, train_l, config, seed)

    cv_results_path = out_dir / "cv_results.csv"
    cv_df.to_csv(cv_results_path, index=False)
    print(f"CV results saved → {cv_results_path}")
    print()

    summary = (
        cv_df.groupby(["feature_set", "classifier"])["macro_f1"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "macro_f1_mean", "std": "macro_f1_std"})
        .reset_index()
        .sort_values("macro_f1_mean", ascending=False)
    )
    print("CV summary (sorted by macro-F1):")
    print(summary.to_string(index=False))
    print()

    # --- Two-pass val selection: macro-F1 then RAG F1 ---
    print("Ranking all grid cells on validation set …")
    all_cells = rank_all_on_val(train_q, train_t, train_l, val_q, val_t, val_l, config, seed)
    finalists = all_cells[:_N_FINALISTS]

    print(f"  Top-{_N_FINALISTS} finalists by val macro-F1:")
    for i, (_, _, meta) in enumerate(finalists):
        print(f"    {i + 1}. {meta['feature_set']}+{meta['classifier_name']}  "
              f"macro-F1={meta['val_macro_f1']:.4f}")
    print()

    # Load raw val rows aligned with val_q for RAG re-ranking
    val_rows_raw = read_jsonl(splits_dir / "val.jsonl")
    oracle_by_id_val: dict[str, int] = {r["qa_id"]: r["best_size"] for r in read_jsonl(oracle_path)}
    val_rows_filtered = [
        r for r in val_rows_raw if oracle_by_id_val.get(r["qa_id"]) in chunk_sizes
    ]

    print("  Re-ranking finalists by val RAG F1 …")
    best_ext, best_clf, best_meta = finalists[0]
    rag_results: list[tuple[float, int]] = []
    for i, (ext, clf, meta) in enumerate(finalists):
        rag_f1 = _val_rag_f1(ext, clf, val_q, val_t, val_rows_filtered, config)
        if rag_f1 is not None:
            rag_results.append((rag_f1, i))
            print(f"    {i + 1}. {meta['feature_set']}+{meta['classifier_name']}  "
                  f"val RAG F1={rag_f1:.4f}")
        else:
            print(f"    {i + 1}. {meta['feature_set']}+{meta['classifier_name']}  "
                  f"val RAG F1=unavailable")

    if rag_results:
        best_rag_f1, best_idx = max(rag_results)
        best_ext, best_clf, best_meta = finalists[best_idx]
        best_meta = {**best_meta, "val_rag_f1": best_rag_f1, "selection_method": "val_rag_f1"}
        print(f"\n  Selected by val RAG F1: "
              f"{best_meta['feature_set']}+{best_meta['classifier_name']} "
              f"(RAG F1={best_rag_f1:.4f})")
    else:
        best_meta = {**best_meta, "selection_method": "val_macro_f1"}
        print("\n  RAG pipeline unavailable — selected by val macro-F1.")
    print()

    pkl_path = out_dir / "best.pkl"
    with pkl_path.open("wb") as f:
        pickle.dump({"extractor": best_ext, "classifier": best_clf, "config": best_meta}, f)
    print(f"Best model saved     → {pkl_path}")

    config_json_path = out_dir / "best_config.json"
    with config_json_path.open("w") as f:
        json.dump(best_meta, f, indent=2)
    print(f"Best config saved    → {config_json_path}")
    print()

    print("Winner:")
    print(f"  feature_set          : {best_meta['feature_set']}")
    print(f"  classifier           : {best_meta['classifier_name']}")
    print(f"  selection method     : {best_meta['selection_method']}")
    print(f"  val macro-F1         : {best_meta['val_macro_f1']:.4f}")
    if "val_rag_f1" in best_meta:
        print(f"  val RAG F1           : {best_meta['val_rag_f1']:.4f}")
    print(f"  val balanced acc     : {best_meta['val_balanced_accuracy']:.4f}")
    print(f"  val accuracy         : {best_meta['val_accuracy']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the chunk-size router.")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    main(load_config(args.config), args.config)
