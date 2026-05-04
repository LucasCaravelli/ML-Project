"""Generate router-specific figures for the report.

Reads from rag-chunk-routing/:
  artifacts/baselines/oracle_gap.json
  artifacts/oracle/eval_grid.jsonl
  results/runs/*_router/metrics.json        (trained router)
  results/runs/*_type_router/metrics.json   (type-aware heuristic, optional)
  results/runs/*_router/predictions.jsonl   (for per-type breakdown)
  results/runs/*_type_router/predictions.jsonl

Produces in results/figures/:
  fig_router_comparison.pdf(.png)   bar chart: Oracle / Type-aware / Baseline / Router
  fig_router_per_type.pdf(.png)     per-type F1 breakdown

Run from rag-chunk-routing/:
    python experiments/make_router_figures.py --config configs/base.yaml
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rag_cr import Config, get_logger, load_config
from rag_cr.io import read_jsonl

log = get_logger(__name__)

TYPES_ORDER = ["factoid", "multihop", "synthesis"]

_RUN_RE = re.compile(r"^\d{8}_\d{6}_(.+)$")

PALETTE = {
    "Oracle\n(ceiling)": "#27ae60",
    "Type-aware\nheuristic": "#f39c12",
    "Best baseline\n(size 128)": "#3498db",
    "Trained router\n(MiniLM+LR)": "#e74c3c",
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _latest(results_dir: Path, tag: str, filename: str) -> Path | None:
    candidates = [
        p for p in sorted(results_dir.glob(f"runs/*_{tag}/{filename}"))
        if (m := _RUN_RE.match(p.parent.name)) and m.group(1) == tag
    ]
    return candidates[-1] if candidates else None


def _figure_router_comparison(out_dir: Path, artifacts_dir: Path, results_dir: Path) -> None:
    oracle_gap_path = artifacts_dir / "baselines" / "oracle_gap.json"
    if not oracle_gap_path.exists():
        log.warning("oracle_gap.json not found — skipping comparison figure")
        return

    oracle_gap = _load_json(oracle_gap_path)
    oracle_f1: float = oracle_gap["oracle_f1_mean"]
    baseline_f1: float = oracle_gap["best_baseline_f1"]

    router_path = _latest(results_dir, "router", "metrics.json")
    if not router_path:
        log.warning("No router run found — skipping comparison figure")
        return
    router_f1: float = _load_json(router_path)["mean_f1"]

    type_router_path = _latest(results_dir, "type_router", "metrics.json")
    type_router_f1: float | None = (
        _load_json(type_router_path)["mean_f1"] if type_router_path else None
    )

    entries: list[tuple[str, float]] = [("Oracle\n(ceiling)", oracle_f1)]
    if type_router_f1 is not None:
        entries.append(("Type-aware\nheuristic", type_router_f1))
    entries.append(("Best baseline\n(size 128)", baseline_f1))
    entries.append(("Trained router\n(MiniLM+LR)", router_f1))

    labels = [e[0] for e in entries]
    values = [e[1] for e in entries]
    colors = [PALETTE.get(lbl, "#95a5a6") for lbl in labels]

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.004,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.axhline(baseline_f1, color="#3498db", linestyle="--", alpha=0.35, linewidth=1.0)
    ax.set_ylabel("Mean F1 (test, $n=84$)", fontsize=10)
    ax.set_ylim(0, oracle_f1 * 1.22)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    for suffix in (".pdf", ".png"):
        out = out_dir / f"fig_router_comparison{suffix}"
        kw = {"bbox_inches": "tight"} if suffix == ".pdf" else {"dpi": 200, "bbox_inches": "tight"}
        fig.savefig(out, **kw)
    plt.close(fig)
    log.info("Saved → %s/fig_router_comparison.*", out_dir)


def _figure_router_per_type(out_dir: Path, artifacts_dir: Path, results_dir: Path) -> None:
    oracle_gap_path = artifacts_dir / "baselines" / "oracle_gap.json"
    eval_grid_path = artifacts_dir / "oracle" / "eval_grid.jsonl"

    if not oracle_gap_path.exists() or not eval_grid_path.exists():
        log.warning("Missing oracle_gap.json or eval_grid.jsonl — skipping per-type figure")
        return

    oracle_gap = _load_json(oracle_gap_path)
    per_type_oracle: dict = oracle_gap.get("per_type", {})

    # Map qa_id -> type from eval_grid
    type_by_id: dict[str, str] = {}
    for row in read_jsonl(eval_grid_path):
        if row.get("split") == "test":
            type_by_id[row["qa_id"]] = row.get("type", "factoid")

    def _per_type_f1(preds_path: Path | None, type_key: str = "question_type") -> dict[str, float]:
        if not preds_path:
            return {}
        by_type: dict[str, list[float]] = defaultdict(list)
        for row in read_jsonl(preds_path):
            qtype = row.get(type_key) or type_by_id.get(row.get("qa_id", ""), "factoid")
            if "f1" in row:
                by_type[qtype].append(float(row["f1"]))
        return {t: float(np.mean(fs)) for t, fs in by_type.items() if fs}

    router_by_type = _per_type_f1(_latest(results_dir, "router", "predictions.jsonl"))
    type_router_by_type = _per_type_f1(_latest(results_dir, "type_router", "predictions.jsonl"))

    x = np.arange(len(TYPES_ORDER))
    width = 0.22

    fig, ax = plt.subplots(figsize=(7.0, 4.0))

    # Best fixed baseline per type
    baseline_vals = [per_type_oracle.get(t, {}).get("best_baseline_f1", 0.0) for t in TYPES_ORDER]
    ax.bar(x - width, baseline_vals, width, label="Best fixed baseline", color="#3498db", alpha=0.85)

    # Type-aware policy
    if type_router_by_type:
        ta_vals = [type_router_by_type.get(t, 0.0) for t in TYPES_ORDER]
        ax.bar(x, ta_vals, width, label="Type-aware heuristic", color="#f39c12", alpha=0.85)
        offset = width
    else:
        offset = 0.0

    # Trained router
    if router_by_type:
        router_vals = [router_by_type.get(t, 0.0) for t in TYPES_ORDER]
        ax.bar(x + offset, router_vals, width, label="Trained router (MiniLM+LR)", color="#e74c3c", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in TYPES_ORDER], fontsize=10)
    ax.set_ylabel("Mean F1", fontsize=10)
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    for suffix in (".pdf", ".png"):
        out = out_dir / f"fig_router_per_type{suffix}"
        kw = {"bbox_inches": "tight"} if suffix == ".pdf" else {"dpi": 200, "bbox_inches": "tight"}
        fig.savefig(out, **kw)
    plt.close(fig)
    log.info("Saved → %s/fig_router_per_type.*", out_dir)


def main(config: Config) -> None:
    artifacts_dir = config.paths.artifacts_dir
    results_dir = Path(config.paths.results_dir)
    out_dir = results_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    _figure_router_comparison(out_dir, artifacts_dir, results_dir)
    _figure_router_per_type(out_dir, artifacts_dir, results_dir)

    log.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    main(load_config(args.config))
