"""Stratify the validated QA set into deterministic train / val / test splits.

Reads the validated QA JSONL, draws a stratified split using config.splits.ratios
(seeded), and writes one JSONL per split plus a manifest with seed, hashes, and
counts. Refuses to overwrite existing splits without --force; preserves any
prior manifest note unless --note is given.

Run from rag-chunk-routing/:
    python experiments/make_splits.py --config configs/base.yaml [--force] [--note "..."]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

from rag_cr import Config, get_logger, load_config, set_seed
from rag_cr.io import (
    ensure_dir,
    get_qa_validated_path,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from rag_cr.splits import SPLIT_NAMES, make_splits
from rag_cr.types import QAPair


def _splits_dir(artifacts_dir: Path) -> Path:
    return artifacts_dir / "splits"


def _split_path(artifacts_dir: Path, name: str) -> Path:
    return _splits_dir(artifacts_dir) / f"{name}.jsonl"


def _manifest_path(artifacts_dir: Path) -> Path:
    return _splits_dir(artifacts_dir) / "manifest.json"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def main(config: Config, force: bool, note: str | None) -> None:
    """Generate stratified train / val / test splits and persist a manifest."""
    set_seed(config.project.seed)
    log = get_logger(__name__)

    artifacts_dir = config.paths.artifacts_dir
    qa_path = get_qa_validated_path(artifacts_dir)
    splits_dir = _splits_dir(artifacts_dir)
    manifest_path = _manifest_path(artifacts_dir)

    existing = [p for name in SPLIT_NAMES if (p := _split_path(artifacts_dir, name)).exists()]
    if existing and not force:
        log.error(
            "Refusing to overwrite existing splits: %s. Pass --force to regenerate.",
            [str(p) for p in existing],
        )
        sys.exit(2)

    rows = read_jsonl(qa_path)
    qa_pairs: list[QAPair] = [r for r in rows]  # type: ignore[misc]
    log.info("Loaded %d validated QA pairs from %s", len(qa_pairs), qa_path)

    splits = make_splits(
        qa_pairs,
        ratios=config.splits.ratios,
        seed=config.project.seed,
        stratify_by=config.splits.stratify_by,
    )

    ensure_dir(splits_dir)
    for name in SPLIT_NAMES:
        tagged = [{**qa, "split": name} for qa in splits[name]]
        write_jsonl(_split_path(artifacts_dir, name), tagged)
        log.info(
            "  %s: %d rows, type counts=%s",
            name,
            len(tagged),
            dict(Counter(qa["type"] for qa in splits[name])),
        )

    # Preserve any audit-trail note on regeneration: --note overrides, otherwise
    # we carry forward the previous manifest's note. Without this, the audit
    # trail (e.g. why the QA total is 398 not 400) would silently disappear
    # whenever someone re-ran make_splits with --force.
    preserved_note: str | None = note
    if preserved_note is None and manifest_path.exists():
        try:
            prev = read_json(manifest_path)
            if isinstance(prev, dict) and isinstance(prev.get("note"), str):
                preserved_note = prev["note"]
        except (OSError, ValueError) as exc:
            log.warning("Could not read previous manifest note from %s: %s", manifest_path, exc)

    manifest: dict[str, object] = {
        "seed": config.project.seed,
        "ratios": config.splits.ratios,
        "stratify_by": config.splits.stratify_by,
        # Use POSIX separators so the manifest is identical across local
        # (Windows) and cluster (Linux) regenerations.
        "qa_validated_path": qa_path.as_posix(),
        "qa_validated_sha256": _sha256_file(qa_path),
        "qa_validated_count": len(qa_pairs),
        "split_counts": {name: len(splits[name]) for name in SPLIT_NAMES},
        "split_type_counts": {
            name: dict(Counter(qa["type"] for qa in splits[name])) for name in SPLIT_NAMES
        },
    }
    if preserved_note is not None:
        manifest["note"] = preserved_note
    write_json(manifest_path, manifest)
    log.info("Wrote manifest → %s", manifest_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--force", action="store_true", help="Overwrite existing splits")
    parser.add_argument(
        "--note",
        default=None,
        help=(
            "Manifest audit-trail note. If omitted, the existing manifest's "
            "note (if any) is preserved across regenerations."
        ),
    )
    args = parser.parse_args()
    main(load_config(args.config), force=args.force, note=args.note)
