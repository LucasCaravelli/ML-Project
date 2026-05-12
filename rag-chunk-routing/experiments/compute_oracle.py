"""Score every (question, chunk_size) cell of the eval grid and emit oracle labels.

For each requested split, retrieves at every chunk size, generates an answer
through the configured backend, scores against gold (EM / F1 / faithfulness),
and append-merges new rows into artifacts/oracle/eval_grid.jsonl. Then derives
labels.jsonl (restricted to the active action space) plus labels_full.jsonl
whenever retired sizes are present.

Run from rag-chunk-routing/:
    python experiments/compute_oracle.py --config configs/base.yaml [--splits test|train|val|all]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from rag_cr import Config, get_logger, load_config, set_seed
from rag_cr.generation import build_prompt, generate_batch
from rag_cr.io import ensure_dir, read_jsonl, write_jsonl
from rag_cr.metrics import score
from rag_cr.oracle import label_from_grid
from rag_cr.retrieval import Retriever
from rag_cr.splits import SPLIT_NAMES


def _split_path(artifacts_dir: Path, name: str) -> Path:
    return artifacts_dir / "splits" / f"{name}.jsonl"


def _eval_grid_path(artifacts_dir: Path) -> Path:
    return artifacts_dir / "oracle" / "eval_grid.jsonl"


def _oracle_labels_path(artifacts_dir: Path) -> Path:
    return artifacts_dir / "oracle" / "labels.jsonl"


def _parse_splits(arg: str) -> list[str]:
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("--splits cannot be empty")
    if "all" in parts:
        return list(SPLIT_NAMES)
    unknown = [p for p in parts if p not in SPLIT_NAMES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown split(s): {unknown}. Valid: {list(SPLIT_NAMES)} or 'all'"
        )
    return parts


def _load_splits(artifacts_dir: Path, names: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in names:
        path = _split_path(artifacts_dir, name)
        if not path.exists():
            raise FileNotFoundError(
                f"Split file not found: {path}. Run experiments/make_splits.py first."
            )
        for row in read_jsonl(path):
            row.setdefault("split", name)
            rows.append(row)
    return rows


def _merge_grid_rows(
    existing: list[dict[str, Any]], new_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Dedup on (qa_id, chunk_size); newer rows overwrite older ones."""
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for row in existing:
        by_key[(row["qa_id"], int(row["chunk_size"]))] = row
    for row in new_rows:
        by_key[(row["qa_id"], int(row["chunk_size"]))] = row
    return [by_key[k] for k in sorted(by_key)]


def main(config: Config, splits: list[str]) -> None:
    """Score the eval grid for the requested splits and emit oracle labels."""
    set_seed(config.project.seed)
    log = get_logger(__name__)

    artifacts_dir = config.paths.artifacts_dir
    sizes = config.chunking.sizes
    k_by_size = {s: config.retrieval.k_for_size(s) for s in sizes}

    qa_rows = _load_splits(artifacts_dir, splits)
    log.info("Loaded %d QA rows across splits=%s", len(qa_rows), splits)

    log.info("Building retriever and prompts (k_by_size=%s, sizes=%s)…", k_by_size, sizes)
    retriever = Retriever(config)
    pending: list[tuple[dict[str, Any], int, list[str], str]] = []
    for qa in qa_rows:
        for size in sizes:
            chunks = retriever.retrieve(qa["question"], size, k_by_size[size])
            passages = [c["text"] for c in chunks]
            prompt = build_prompt(qa["question"], passages, config)
            pending.append((qa, size, passages, prompt))
    log.info("Built %d prompts (qa=%d × sizes=%d)", len(pending), len(qa_rows), len(sizes))

    log.info(
        "Generating with backend=%s model=%s",
        config.generation.backend,
        config.generation.model_name,
    )
    predictions = generate_batch([p[3] for p in pending], config)
    if len(predictions) != len(pending):
        log.error(
            "generate_batch returned %d predictions for %d prompts",
            len(predictions),
            len(pending),
        )
        sys.exit(2)

    log.info("Scoring %d predictions", len(predictions))
    new_rows: list[dict[str, Any]] = []
    for (qa, size, passages, _), pred in zip(pending, predictions, strict=False):
        s = score(pred, qa["answer"], passages)
        new_rows.append(
            {
                "qa_id": qa["qa_id"],
                "chunk_size": size,
                "split": qa["split"],
                "type": qa["type"],
                "question": qa["question"],
                "gold": qa["answer"],
                "prediction": pred,
                "passages": passages,
                "em": s["em"],
                "f1": s["f1"],
                "faithfulness": s["faithfulness"],
                "cost_tokens": s["cost_tokens"],
            }
        )

    grid_path = _eval_grid_path(artifacts_dir)
    ensure_dir(grid_path.parent)
    existing = read_jsonl(grid_path) if grid_path.exists() else []
    merged = _merge_grid_rows(existing, new_rows)
    write_jsonl(grid_path, merged)
    log.info("Eval grid: existing=%d new=%d total=%d → %s",
             len(existing), len(new_rows), len(merged), grid_path)

    # Canonical labels.jsonl is the router's training target, so it must be
    # restricted to the active action space (config.chunking.sizes). Rows for
    # retired sizes (e.g. 1024) stay in eval_grid.jsonl for the ablation but
    # must not leak into labels — otherwise best_size could name a size the
    # router cannot select. Full-grid labels are written to labels_full.jsonl
    # for the ablation.
    active_sizes = set(config.chunking.sizes)
    active_rows = [r for r in merged if int(r["chunk_size"]) in active_sizes]
    labels = label_from_grid(active_rows)
    labels_path = _oracle_labels_path(artifacts_dir)
    write_jsonl(labels_path, [dict(label) for label in labels])
    log.info(
        "Oracle labels (action_space=%s): %d → %s",
        sorted(active_sizes), len(labels), labels_path,
    )

    if len(active_rows) < len(merged):
        labels_full = label_from_grid(merged)
        labels_full_path = labels_path.with_name("labels_full.jsonl")
        write_jsonl(labels_full_path, [dict(label) for label in labels_full])
        log.info(
            "Oracle labels (full grid, %d sizes): %d → %s",
            len({int(r["chunk_size"]) for r in merged}),
            len(labels_full),
            labels_full_path,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument(
        "--splits",
        default="test",
        type=_parse_splits,
        help="Comma-separated split names to score (test|train|val|all). Default: test",
    )
    args = parser.parse_args()
    main(load_config(args.config), splits=args.splits)
