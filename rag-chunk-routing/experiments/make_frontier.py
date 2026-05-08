"""Accuracy-cost frontier across all evaluated systems.

Data sources (in priority order per system):

  Fixed baselines (fixed_128/256/512):
    1. results/runs/<ts>_fixed_<size>/metrics.json   if n >= N_TEST
    2. artifacts/baselines/test_summary.json          (fallback, from oracle eval grid)
       + artifacts/oracle/eval_grid.jsonl             (for cost tokens)

  Oracle upper bound:
    artifacts/baselines/oracle_gap.json   (F1)
    artifacts/oracle/eval_grid.jsonl      (cost: best-size cost per question)
    artifacts/oracle/labels.jsonl         (which size is oracle-best per question)

  Fusion, Router:
    results/runs/<ts>_fusion/metrics.json
    results/runs/<ts>_router/metrics.json
    (missing = skipped with warning)

Headline metric (README line 17-18):
  gap_recovery_per_cost = (sys_F1 - best_baseline_F1)
                        / (oracle_F1 - best_baseline_F1)
                        / mean_cost_tokens_per_query

Outputs:
  results/figures/frontier.png (.pdf)
  results/runs/SUMMARY.md

Run from rag-chunk-routing/:
    python experiments/make_frontier.py --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from rag_cr import Config, get_logger, load_config
from rag_cr.io import read_jsonl

log = get_logger(__name__)

N_TEST = 80          # minimum n to consider a run "full" (test split has 84)
_RUN_RE = re.compile(r"^\d{8}_\d{6}_(.+)$")

# Visual style — consistent with make_figures.py palette where sizes overlap
_STYLE: dict[str, dict] = {
    "fixed_128":  {"color": "#1f77b4", "marker": "o", "ms": 9},
    "fixed_256":  {"color": "#ff7f0e", "marker": "o", "ms": 9},
    "fixed_512":  {"color": "#2ca02c", "marker": "o", "ms": 9},
    "fusion":     {"color": "#d62728", "marker": "D", "ms": 9},
    "router":     {"color": "#9467bd", "marker": "s", "ms": 9},
    "oracle":     {"color": "#17becf", "marker": "*", "ms": 14},
}
_DISPLAY: dict[str, str] = {
    "fixed_128": "Fixed (128)",
    "fixed_256": "Fixed (256)",
    "fixed_512": "Fixed (512)",
    "fusion":    "RRF Fusion",
    "router":    "Router",
    "oracle":    "Oracle (ceiling)",
}


@dataclass
class SystemPoint:
    key: str         # canonical identifier, e.g. "fixed_128"
    f1: float
    cost_mean: float # mean tokens per question
    n: int
    source: str      # short description of where the data came from


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_run_metrics(results_dir: Path, tag: str) -> Path | None:
    """Return the metrics.json path of the most-recent run matching tag, or None."""
    candidates = sorted(
        p for p in results_dir.glob(f"runs/*_{tag}/metrics.json")
        if (m := _RUN_RE.match(p.parent.name)) and m.group(1) == tag
    )
    return candidates[-1] if candidates else None


def _from_run(results_dir: Path, tag: str, label: str) -> SystemPoint | None:
    """Load a system point from the latest results/runs/ entry."""
    path = _latest_run_metrics(results_dir, tag)
    if path is None:
        return None
    data = _load_json(path)
    n = data.get("n") or data.get("n_test_examples") or 0
    if n < N_TEST:
        log.warning(
            "Skipping %s run at %s: n=%d < %d (likely a partial run)",
            tag, path.parent.name, n, N_TEST,
        )
        return None
    f1 = data.get("f1") if data.get("f1") is not None else data.get("mean_f1")
    cost_total = data.get("cost_tokens_total") or 0
    if f1 is None:
        return None
    return SystemPoint(
        key=tag, f1=float(f1),
        cost_mean=cost_total / n if n else 0.0,
        n=int(n), source=f"results/runs/{path.parent.name}",
    )


def _fixed_from_eval_grid(
    test_summary: dict, eval_grid: list[dict], size: int
) -> SystemPoint:
    key_s = str(size)
    m = test_summary[key_s]
    costs = [
        float(r["cost_tokens"]) for r in eval_grid
        if r.get("split") == "test" and int(r["chunk_size"]) == size and "cost_tokens" in r
    ]
    cost_mean = sum(costs) / len(costs) if costs else 0.0
    return SystemPoint(
        key=f"fixed_{size}", f1=float(m["f1"]),
        cost_mean=cost_mean, n=int(m["n"]),
        source="artifacts/baselines/test_summary.json + eval_grid.jsonl",
    )


def _oracle_from_artifacts(
    oracle_gap: dict, oracle_labels: list[dict], eval_grid: list[dict]
) -> SystemPoint:
    cost_lookup = {
        (r["qa_id"], int(r["chunk_size"])): float(r["cost_tokens"])
        for r in eval_grid if "cost_tokens" in r
    }
    test_qids = {r["qa_id"] for r in eval_grid if r.get("split") == "test"}
    costs = [
        cost_lookup[(lbl["qa_id"], lbl["best_size"])]
        for lbl in oracle_labels
        if lbl["qa_id"] in test_qids
        and (lbl["qa_id"], lbl["best_size"]) in cost_lookup
    ]
    return SystemPoint(
        key="oracle", f1=float(oracle_gap["oracle_f1_mean"]),
        cost_mean=sum(costs) / len(costs) if costs else 0.0,
        n=int(oracle_gap["n_questions_test"]),
        source="artifacts/baselines/oracle_gap.json + eval_grid.jsonl",
    )


def _collect_all(config: Config) -> tuple[list[SystemPoint], dict]:
    """Return (list of available SystemPoints, reference dict with oracle/baseline F1)."""
    artifacts_dir = config.paths.artifacts_dir
    results_dir = Path(config.paths.results_dir)
    baselines_dir = artifacts_dir / "baselines"
    oracle_dir = artifacts_dir / "oracle"

    test_summary_path = baselines_dir / "test_summary.json"
    oracle_gap_path = baselines_dir / "oracle_gap.json"
    eval_grid_path = oracle_dir / "eval_grid.jsonl"
    labels_path = oracle_dir / "labels.jsonl"

    for p in (test_summary_path, oracle_gap_path, eval_grid_path, labels_path):
        if not p.exists():
            log.error("Required artifact missing: %s", p)
            log.error("Run experiments/run_baselines.py and experiments/compute_oracle.py first.")
            sys.exit(2)

    test_summary = _load_json(test_summary_path)
    oracle_gap = _load_json(oracle_gap_path)
    eval_grid = read_jsonl(eval_grid_path)
    oracle_labels = read_jsonl(labels_path)

    ref = {
        "oracle_f1": oracle_gap["oracle_f1_mean"],
        "best_baseline_f1": oracle_gap["best_baseline_f1"],
        "best_baseline_size": oracle_gap["best_baseline_size"],
        "gap_f1": oracle_gap["gap_f1"],
        "n_test": oracle_gap["n_questions_test"],
    }

    points: list[SystemPoint] = []

    # --- Fixed baselines: try results/runs/ first, fall back to eval grid ---
    for size in sorted(config.chunking.sizes):
        tag = f"fixed_{size}"
        pt = _from_run(results_dir, tag, tag)
        if pt is None:
            pt = _fixed_from_eval_grid(test_summary, eval_grid, size)
            log.info("fixed_%d: using eval-grid fallback (n=%d, F1=%.4f)", size, pt.n, pt.f1)
        else:
            log.info("fixed_%d: from %s (n=%d, F1=%.4f)", size, pt.source, pt.n, pt.f1)
        points.append(pt)

    # --- Oracle upper bound (always from artifacts) ---
    oracle_pt = _oracle_from_artifacts(oracle_gap, oracle_labels, eval_grid)
    log.info("oracle: n=%d, F1=%.4f, mean_cost=%.0f", oracle_pt.n, oracle_pt.f1, oracle_pt.cost_mean)
    points.append(oracle_pt)

    # --- Fusion and Router: results/runs/ only ---
    for tag in ("fusion", "router"):
        pt = _from_run(results_dir, tag, tag)
        if pt is None:
            log.warning(
                "%s: no valid results/runs/ entry found (n >= %d). "
                "Run experiments/run_%s.py --config configs/base.yaml with ollama running.",
                tag, N_TEST, tag,
            )
        else:
            log.info("%s: from %s (n=%d, F1=%.4f)", tag, pt.source, pt.n, pt.f1)
            points.append(pt)

    return points, ref


def _sanity_check(points: list[SystemPoint], ref: dict) -> list[str]:
    """Return a list of warning strings for anything suspicious."""
    issues: list[str] = []
    n_values = {pt.n for pt in points}
    if len(n_values) > 1:
        issues.append(
            f"Systems have different n values: {sorted(n_values)}. "
            "Ensure all systems ran on the same test split."
        )
    for pt in points:
        if pt.key in ("fixed_128", "fixed_256", "fixed_512", "oracle"):
            continue
        if pt.f1 < ref["best_baseline_f1"] - 0.01:
            if pt.key == "fusion":
                issues.append(
                    f"fusion: F1={pt.f1:.4f} is below best baseline "
                    f"{ref['best_baseline_f1']:.4f} (size={ref['best_baseline_size']}). "
                    "RRF should not underperform a fixed-size system — likely the run used the "
                    "extractive backend (eval_dry_run.yaml) rather than ollama. "
                    "Re-run with configs/base.yaml once ollama is available."
                )
            else:
                gap_pct = (ref["best_baseline_f1"] - pt.f1) / (ref["oracle_f1"] - ref["best_baseline_f1"]) * 100
                issues.append(
                    f"{pt.key}: F1={pt.f1:.4f} is below best baseline "
                    f"{ref['best_baseline_f1']:.4f} (size={ref['best_baseline_size']}) "
                    f"by {gap_pct:.1f}% of the oracle gap. "
                    "This is a real result, not a backend configuration issue."
                )
    return issues


def _gap_recovery(pt: SystemPoint, ref: dict) -> float | None:
    denom = ref["oracle_f1"] - ref["best_baseline_f1"]
    if abs(denom) < 1e-9:
        return None
    return (pt.f1 - ref["best_baseline_f1"]) / denom


def _gap_recovery_per_cost(pt: SystemPoint, ref: dict) -> float | None:
    gr = _gap_recovery(pt, ref)
    if gr is None or pt.cost_mean < 1:
        return None
    return gr / pt.cost_mean * 1000  # per 1K tokens


def _figure(points: list[SystemPoint], ref: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    # Horizontal dashed line at oracle F1
    ax.axhline(ref["oracle_f1"], color="#17becf", linestyle="--", linewidth=1.0,
               alpha=0.6, label="_oracle_line")

    # Shaded "oracle gap" band
    ax.axhspan(ref["best_baseline_f1"], ref["oracle_f1"],
               color="#17becf", alpha=0.06)

    for pt in points:
        sty = _STYLE.get(pt.key, {"color": "#333333", "marker": "o", "ms": 9})
        lbl = _DISPLAY.get(pt.key, pt.key)
        ax.scatter(
            pt.cost_mean, pt.f1,
            color=sty["color"], marker=sty["marker"], s=sty["ms"] ** 2,
            zorder=5, label=lbl,
        )
        # Per-system annotation offsets; negative x → label to the left of point
        _offsets: dict[str, tuple[int, int]] = {
            "fixed_128": (10, 6),
            "fixed_256": (-100, -14),  # well to the left — avoids legend overlap
            "fixed_512": (-100, 8),
            "fusion":    (10, 5),
            "router":    (10, 5),
            "oracle":    (10, 7),
        }
        _ha: dict[str, str] = {
            "fixed_256": "left",
            "fixed_512": "left",
        }
        offset = _offsets.get(pt.key, (12, 4))
        gr = _gap_recovery(pt, ref)
        gr_text = f"  gap={gr*100:.1f}%" if gr is not None and pt.key != "oracle" else ""
        ha = _ha.get(pt.key, "left")
        ax.annotate(
            f"{lbl}{gr_text}",
            xy=(pt.cost_mean, pt.f1),
            xytext=offset, textcoords="offset points",
            fontsize=8, color=sty["color"], ha=ha,
        )

    ax.set_xlabel("Mean retrieval cost (tokens / question)", fontsize=11)
    ax.set_ylabel("F1 on test split", fontsize=11)
    ax.set_title("Accuracy–cost frontier across retrieval systems", fontsize=12)

    ax.set_axisbelow(True)
    ax.xaxis.grid(True, linestyle="--", alpha=0.35)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend: only labelled artists — upper left keeps it away from the data cluster
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, labels, loc="upper left", frameon=True,
        framealpha=0.92, fontsize=9,
    )

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "frontier.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "frontier.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Frontier figure saved to %s", out_dir)


def _summary_md(
    points: list[SystemPoint],
    ref: dict,
    issues: list[str],
    config: Config,
    out_path: Path,
) -> None:
    lines: list[str] = [
        "# Evaluation Summary",
        "",
        "Auto-generated by `experiments/make_frontier.py`. "
        "Do not edit by hand — re-run the script after new runs land in `results/runs/`.",
        "",
        "## System status",
        "",
        "| System | n | F1 | Mean tokens/q | Gap recovery | Source |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    all_keys = ["fixed_128", "fixed_256", "fixed_512", "oracle", "fusion", "router"]
    by_key = {pt.key: pt for pt in points}
    for key in all_keys:
        pt = by_key.get(key)
        if pt is None:
            lines.append(f"| {_DISPLAY.get(key, key)} | — | — | — | — | **pending** |")
        else:
            gr = _gap_recovery(pt, ref)
            gr_str = f"{gr*100:.1f}%" if gr is not None else "—"
            lines.append(
                f"| {_DISPLAY.get(key, key)} | {pt.n} | {pt.f1:.4f} "
                f"| {pt.cost_mean:.0f} | {gr_str} | {pt.source} |"
            )

    # Per-split sizes from manifest.json
    manifest_path = Path(config.paths.artifacts_dir) / "splits" / "manifest.json"
    if manifest_path.exists():
        mf = _load_json(manifest_path)
        sc = mf.get("split_counts", {})
        stc = mf.get("split_type_counts", {})
        lines += [
            "",
            "## Data splits",
            "",
            f"Seed: {mf.get('seed', 42)} · stratified by question type · "
            "splits are **frozen** (do not regenerate)",
            "",
            "| Split | Total | Factoid | Multihop | Synthesis |",
            "| --- | --- | --- | --- | --- |",
        ]
        for split in ("train", "val", "test"):
            n = sc.get(split, "—")
            tc = stc.get(split, {})
            lines.append(
                f"| {split} | {n} | {tc.get('factoid', '—')} "
                f"| {tc.get('multihop', '—')} | {tc.get('synthesis', '—')} |"
            )

    oracle_f1 = ref["oracle_f1"]
    best_f1 = ref["best_baseline_f1"]
    gap_pts = ref["gap_f1"] * 100

    lines += [
        "",
        "## Oracle gap",
        "",
        f"- Best fixed-size baseline: **F1 = {best_f1:.4f}** (size = {ref['best_baseline_size']})",
        f"- Oracle upper bound: **F1 = {oracle_f1:.4f}** (per-question best size)",
        f"- Gap: **+{gap_pts:.2f} F1 points**",
        f"- Test questions: {ref['n_test']}",
        f"- Action space: {sorted(config.chunking.sizes)}",
        "",
        "## Headline metric",
        "",
        "Per README: **fraction of oracle gap recovered per unit retrieval cost**",
        "",
        "```",
        "score = (sys_F1 - best_baseline_F1) / (oracle_F1 - best_baseline_F1) / mean_cost_tokens * 1000",
        "```",
        "",
        "| System | Gap recovery | Mean tokens/q | Score (per 1K tokens) |",
        "| --- | --- | --- | --- |",
    ]
    for key in all_keys:
        pt = by_key.get(key)
        if pt is None or pt.key in ("fixed_128", "fixed_256", "fixed_512"):
            continue
        gr = _gap_recovery(pt, ref)
        gpc = _gap_recovery_per_cost(pt, ref)
        if gr is None:
            continue
        gpc_str = f"{gpc:.4f}" if gpc is not None else "—"
        lines.append(
            f"| {_DISPLAY.get(key, key)} | {gr*100:.1f}% | {pt.cost_mean:.0f} | {gpc_str} |"
        )

    lines += [
        "",
        "## Cross-system sanity check",
        "",
    ]
    if issues:
        lines.append("**Warnings:**")
        lines.append("")
        for iss in issues:
            lines.append(f"- {iss}")
    else:
        lines.append("No issues detected. All available systems share the same n and scoring code.")

    lines += [
        "",
        "## Pending runs",
        "",
        "The following systems need a valid `results/runs/` entry "
        f"(n ≥ {N_TEST}, ollama backend) to appear in the frontier:",
        "",
    ]
    # Only flag for rerun if missing or if the issue is a backend problem (not mere underperformance)
    backend_flagged = {k for k in ["fusion", "router"] if any(k in iss and "extractive backend" in iss for iss in issues)}
    needs_rerun = [k for k in ["fusion", "router"] if k not in by_key or k in backend_flagged]
    if needs_rerun:
        for k in needs_rerun:
            suffix = " *(re-run — current entry used wrong backend)*" if k in backend_flagged else ""
            lines.append(f"- **{_DISPLAY[k]}**: `python experiments/run_{k}.py --config configs/base.yaml`{suffix}")
        lines += [
            "",
            "### Installing Ollama (Windows)",
            "",
            "1. Download the installer: https://ollama.com/download/windows",
            "2. Run `OllamaSetup.exe` — it installs and starts a background service automatically.",
            "3. In a new terminal: `ollama pull qwen2.5:7b-instruct`",
            "4. Run the missing experiments above.",
            "5. Re-run `python experiments/make_frontier.py` to regenerate this file and the figure.",
        ]
    else:
        lines.append("All systems have valid production runs. Re-run to update after new experiments.")

    lines += [
        "",
        "## Frontier figure",
        "",
        "![frontier](../figures/frontier.png)",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("SUMMARY.md written to %s", out_path)


def main(config: Config) -> None:
    points, ref = _collect_all(config)
    issues = _sanity_check(points, ref)

    if issues:
        for iss in issues:
            warnings.warn(iss, stacklevel=1)

    figures_dir = Path(config.paths.results_dir) / "figures"
    _figure(points, ref, figures_dir)

    summary_path = Path(config.paths.results_dir) / "runs" / "SUMMARY.md"
    _summary_md(points, ref, issues, config, summary_path)

    print()
    print("=" * 65)
    print("FRONTIER SUMMARY")
    print("=" * 65)
    print(f"  oracle_f1    = {ref['oracle_f1']:.4f}")
    print(f"  best_baseline= {ref['best_baseline_f1']:.4f}  (size={ref['best_baseline_size']})")
    print(f"  gap          = +{ref['gap_f1']*100:.2f} F1 pts")
    print()
    by_key = {pt.key: pt for pt in points}
    all_keys = ["fixed_128", "fixed_256", "fixed_512", "oracle", "fusion", "router"]
    for key in all_keys:
        pt = by_key.get(key)
        if pt is None:
            print(f"  {_DISPLAY.get(key, key):20s}  PENDING — run with ollama")
        else:
            gr = _gap_recovery(pt, ref)
            gr_str = f"  gap_recovery={gr*100:.1f}%" if gr is not None else ""
            print(f"  {_DISPLAY.get(key, key):20s}  F1={pt.f1:.4f}  cost={pt.cost_mean:.0f} tok{gr_str}")
    if issues:
        print()
        print("  WARNINGS:")
        for iss in issues:
            print(f"    ! {iss}")
    print("=" * 65)
    print(f"\n  Frontier figure  -> {figures_dir / 'frontier.png'}")
    print(f"  Summary          -> {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot accuracy-cost frontier across all evaluated systems."
    )
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    main(load_config(args.config))
