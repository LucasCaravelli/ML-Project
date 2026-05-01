from __future__ import annotations

import argparse
import json
import pickle
import subprocess
from pathlib import Path

from rag_cr import Config, load_config, set_seed
from rag_cr.router.train import (
    load_router_data,
    run_cv_grid,
    select_best_on_val,
)


def _git_hash() -> str:
    """Return the current git commit hash, or 'unknown' on failure."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _oracle_labels_path(config: Config) -> Path:
    return config.paths.artifacts_dir / "oracle" / "labels.jsonl"


def main(config: Config, config_path: str) -> None:
    """Orchestrate router training: CV grid search, val selection, artifact saving."""
    set_seed(config.project.seed)
    seed = config.project.seed

    print(f"git commit : {_git_hash()}")
    print(f"config     : {config_path}")
    print()

    artifacts_dir = config.paths.artifacts_dir
    splits_dir = artifacts_dir / "splits"
    oracle_path = _oracle_labels_path(config)
    out_dir = artifacts_dir / "router"
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk_sizes: list[int] = [s for s in config.chunking.sizes if s != 1024]
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

    # --- Select best cell on val ---
    print("Selecting best configuration on validation set …")
    best_ext, best_clf, best_meta = select_best_on_val(
        train_q, train_t, train_l,
        val_q, val_t, val_l,
        config, seed,
    )

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
    print(f"  val macro-F1         : {best_meta['val_macro_f1']:.4f}")
    print(f"  val balanced acc     : {best_meta['val_balanced_accuracy']:.4f}")
    print(f"  val accuracy         : {best_meta['val_accuracy']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the chunk-size router.")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    main(load_config(args.config), args.config)
