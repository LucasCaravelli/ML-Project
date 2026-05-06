from __future__ import annotations

import argparse
import json
import subprocess

from rag_cr import Config, load_config, set_seed
from rag_cr import pipeline as _pipeline


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main(config: Config, config_path: str) -> None:
    set_seed(config.project.seed)

    test_path = config.paths.artifacts_dir / "splits" / "test.jsonl"
    if not test_path.exists():
        raise FileNotFoundError(f"Test split not found at {test_path}")

    print(f"git commit : {_git_hash()}")
    print(f"config     : {config_path}")
    print(f"test split : {test_path}")
    print()

    run_dir = _pipeline.run("fusion", test_path, config)

    # --- Sanity check: fusion vs best fixed-size baseline ---
    oracle_gap_path = config.paths.artifacts_dir / "baselines" / "oracle_gap.json"
    metrics = json.loads((run_dir / "metrics.json").read_text())
    fusion_f1 = metrics["f1"]

    print()
    print("=" * 60)
    print("FUSION SANITY CHECK")
    print("=" * 60)
    print(f"  fusion  n={metrics['n']}  F1={fusion_f1:.4f}  "
          f"EM={metrics['em']:.4f}  tokens={metrics['cost_tokens_total']}")

    if oracle_gap_path.exists():
        gap_data = json.loads(oracle_gap_path.read_text())
        best_f1 = gap_data.get("best_baseline_f1", 0.0)
        best_size = gap_data.get("best_baseline_size", "?")
        oracle_f1 = gap_data.get("oracle_f1_mean")
        delta = fusion_f1 - best_f1
        print(f"  fixed_{best_size}          F1={best_f1:.4f}  (best baseline)")
        if oracle_f1 is not None:
            print(f"  oracle              F1={oracle_f1:.4f}  (upper bound)")
        print()
        if config.generation.backend != "ollama":
            print(f"  NOTE: backend={config.generation.backend!r} — "
                  "baseline was computed with ollama; comparison is not apples-to-apples")
        elif delta < 0:
            print(f"  WARNING: fusion is {abs(delta) * 100:.2f} F1 pts BELOW best baseline "
                  "— check RRF wiring in retrieval.py._retrieve_fusion()")
        else:
            print(f"  OK: fusion beats best baseline by +{delta * 100:.2f} F1 pts "
                  "(higher token cost is expected)")
    else:
        print("  (oracle_gap.json not found — skipping baseline comparison)")

    print("=" * 60)
    print(f"\nRun dir: {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RRF fusion system on test split.")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    main(load_config(args.config), args.config)
