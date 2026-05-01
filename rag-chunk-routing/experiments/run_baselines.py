from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rag_cr import Config, get_logger, load_config
from rag_cr.io import read_jsonl, write_json


def _eval_grid_path(artifacts_dir: Path) -> Path:
    return artifacts_dir / "oracle" / "eval_grid.jsonl"


def _labels_path(artifacts_dir: Path) -> Path:
    return artifacts_dir / "oracle" / "labels.jsonl"


def _baselines_dir(artifacts_dir: Path) -> Path:
    return artifacts_dir / "baselines"


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _per_size_baseline(test_rows: list[dict]) -> dict[int, dict[str, float]]:
    by_size: dict[int, list[dict]] = defaultdict(list)
    for row in test_rows:
        by_size[int(row["chunk_size"])].append(row)
    return {
        size: {
            "n": len(rows),
            "f1": _mean([float(r["f1"]) for r in rows]),
            "em": _mean([float(r["em"]) for r in rows]),
            "faithfulness": _mean([float(r["faithfulness"]) for r in rows]),
        }
        for size, rows in sorted(by_size.items())
    }


def _oracle_f1_per_qa(test_rows: list[dict]) -> dict[str, float]:
    by_qa: dict[str, float] = {}
    for row in test_rows:
        f1 = float(row["f1"])
        prev = by_qa.get(row["qa_id"])
        if prev is None or f1 > prev:
            by_qa[row["qa_id"]] = f1
    return by_qa


def _per_type_gap(test_rows: list[dict]) -> dict[str, dict[str, float]]:
    by_type_qa: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in test_rows:
        by_type_qa[row["type"]][row["qa_id"]].append(row)

    out: dict[str, dict[str, float]] = {}
    for qtype, qa_map in by_type_qa.items():
        oracle_f1 = [max(float(r["f1"]) for r in rows) for rows in qa_map.values()]
        per_size: dict[int, list[float]] = defaultdict(list)
        for rows in qa_map.values():
            for r in rows:
                per_size[int(r["chunk_size"])].append(float(r["f1"]))
        per_size_mean = {size: _mean(scores) for size, scores in per_size.items()}
        best_baseline = max(per_size_mean.values()) if per_size_mean else 0.0
        oracle_mean = _mean(oracle_f1)
        out[qtype] = {
            "n_questions": len(qa_map),
            "oracle_f1": oracle_mean,
            "best_baseline_f1": best_baseline,
            "gap": oracle_mean - best_baseline,
            "per_size_f1": per_size_mean,
        }
    return out


def _size_distribution(rows_for_split: list[dict]) -> dict[int, dict[str, float]]:
    by_qa: dict[str, list[dict]] = defaultdict(list)
    for row in rows_for_split:
        by_qa[row["qa_id"]].append(row)

    best_sizes: list[int] = []
    for rows in by_qa.values():
        winner = min(
            rows,
            key=lambda r: (-float(r["f1"]), -float(r["em"]), int(r["chunk_size"])),
        )
        best_sizes.append(int(winner["chunk_size"]))

    n = len(best_sizes)
    counts = Counter(best_sizes)
    return {
        size: {"count": counts.get(size, 0), "share": counts.get(size, 0) / n if n else 0.0}
        for size in sorted({s for r in rows_for_split for s in [int(r["chunk_size"])]})
    }


def _verdict(gap: float, max_share: float) -> str:
    if gap >= 3.0 and max_share < 0.40:
        return "HEALTHY -- oracle gap is real and size distribution is non-trivial. Proceed to Stage B."
    if gap < 1.0 or max_share > 0.60:
        return ("PIVOT -- gap too small or one chunk size dominates. "
                "The premise is weaker than assumed; do not run Stage B. "
                "Discuss with the team before any router work.")
    return ("MARGINAL -- gap and/or distribution are borderline. "
            "Re-evaluate the project framing with the team before committing to router work.")


def _parse_sizes(arg: str) -> list[int]:
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("--restrict-sizes cannot be empty")
    try:
        sizes = [int(p) for p in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--restrict-sizes must be a comma-separated list of integers, got {arg!r}"
        ) from exc
    return sizes


def main(
    config: Config,
    restrict_sizes: list[int] | None = None,
    out_suffix: str = "",
) -> None:
    log = get_logger(__name__)
    artifacts_dir = config.paths.artifacts_dir

    grid_path = _eval_grid_path(artifacts_dir)
    if not grid_path.exists():
        raise FileNotFoundError(
            f"Eval grid not found at {grid_path}. Run experiments/compute_oracle.py first."
        )

    grid = read_jsonl(grid_path)
    log.info("Loaded %d eval-grid rows from %s", len(grid), grid_path)

    # Default: report on the active action space declared in chunking.sizes.
    # Anything else in the eval grid (e.g. the retired size=1024 rows that
    # remain on disk for the ablation) is filtered out so the canonical
    # baselines / oracle gap reflect what the router will actually see.
    # Pass --restrict-sizes to override (e.g. include 1024 for an ablation).
    allowed = set(restrict_sizes) if restrict_sizes is not None else set(config.chunking.sizes)
    before = len(grid)
    grid = [r for r in grid if int(r["chunk_size"]) in allowed]
    log.info(
        "Action space sizes=%s: %d -> %d rows",
        sorted(allowed), before, len(grid),
    )

    test_rows = [r for r in grid if r.get("split") == "test"]
    if not test_rows:
        log.warning("No rows with split='test' in eval grid; gap will be skipped.")

    out_dir = _baselines_dir(artifacts_dir)

    def _named(stem: str) -> Path:
        return out_dir / f"{stem}{out_suffix}.json"

    # 1. Per-size baseline summary on test
    per_size = _per_size_baseline(test_rows)
    write_json(_named("test_summary"), per_size)

    # 2. Size distribution per split
    by_split: dict[str, list[dict]] = defaultdict(list)
    for r in grid:
        by_split[r.get("split", "unknown")].append(r)
    size_dist = {split: _size_distribution(rows) for split, rows in sorted(by_split.items())}
    write_json(_named("size_distribution"), size_dist)

    # 3. Oracle gap on test (overall + per-type)
    oracle_gap_payload: dict[str, Any] = {}
    if test_rows:
        oracle_per_qa = _oracle_f1_per_qa(test_rows)
        oracle_mean_test = _mean(list(oracle_per_qa.values()))
        best_size_baseline = max(per_size.values(), key=lambda v: v["f1"]) if per_size else None
        best_baseline_f1 = best_size_baseline["f1"] if best_size_baseline else 0.0
        gap_f1 = oracle_mean_test - best_baseline_f1
        oracle_gap_payload = {
            "action_space": sorted(allowed),
            "n_questions_test": len(oracle_per_qa),
            "oracle_f1_mean": oracle_mean_test,
            "best_baseline_f1": best_baseline_f1,
            "best_baseline_size": next(
                (s for s, v in per_size.items() if v["f1"] == best_baseline_f1), None
            ),
            "gap_f1": gap_f1,
            "gap_f1_points": gap_f1 * 100,
            "per_type": _per_type_gap(test_rows),
        }
        write_json(_named("oracle_gap"), oracle_gap_payload)

    # ---- Pretty-printed report ----
    print()
    print("=" * 72)
    print("BASELINES + ORACLE GAP REPORT")
    print("=" * 72)

    print("\n[Per-size baselines on test]")
    if per_size:
        for size, m in per_size.items():
            print(f"  size={size:>4}  n={m['n']:>3}  F1={m['f1']:.4f}  EM={m['em']:.4f}  faith={m['faithfulness']:.4f}")
    else:
        print("  (no test rows)")

    print("\n[Oracle-best size distribution]")
    for split, dist in size_dist.items():
        breakdown = ", ".join(f"{s}={d['share']:.1%}" for s, d in dist.items())
        print(f"  {split:>5}: {breakdown}")

    if test_rows:
        gap_pts = oracle_gap_payload["gap_f1_points"]
        best_size = oracle_gap_payload["best_baseline_size"]
        print("\n[Oracle gap on test]")
        print(f"  oracle F1 (mean per-qa max):     {oracle_gap_payload['oracle_f1_mean']:.4f}")
        print(f"  best fixed-size baseline F1:     {oracle_gap_payload['best_baseline_f1']:.4f}  (size={best_size})")
        print(f"  GAP:                             {gap_pts:+.2f} F1 points")

        print("\n[Per-type oracle gap on test]")
        for qtype, m in oracle_gap_payload["per_type"].items():
            print(f"  {qtype:>10}  n={m['n_questions']:>3}  oracle={m['oracle_f1']:.4f}  best={m['best_baseline_f1']:.4f}  gap={m['gap']*100:+.2f} pts")

        # Verdict using the test-split distribution if available, else first split.
        max_share = max(
            (d["share"] for d in size_dist.get("test", next(iter(size_dist.values()), {})).values()),
            default=1.0,
        )
        verdict = _verdict(gap_pts, max_share)
        print("\n[Verdict]")
        print(f"  gap={gap_pts:+.2f} pts   max single-size share (test)={max_share:.1%}")
        print(f"  -> {verdict}")

    print("\n" + "=" * 72)
    print(f"Wrote: {out_dir}")
    print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument(
        "--restrict-sizes",
        type=_parse_sizes,
        default=None,
        help=(
            "Comma-separated list of chunk sizes to keep when aggregating "
            "(e.g. '128,256,512,1024' to include the retired 1024 rows for an "
            "ablation). Default: use chunking.sizes from the config, which is "
            "the active router action space."
        ),
    )
    parser.add_argument(
        "--out-suffix",
        default="",
        help=(
            "Suffix appended to the output JSON filenames (e.g. '_full' "
            "produces oracle_gap_full.json). Use this to keep ablation "
            "outputs alongside the canonical reports without overwriting."
        ),
    )
    args = parser.parse_args()
    main(load_config(args.config), restrict_sizes=args.restrict_sizes, out_suffix=args.out_suffix)
