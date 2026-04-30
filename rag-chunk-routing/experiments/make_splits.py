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


def main(config: Config, force: bool) -> None:
    set_seed(config.project.seed)
    log = get_logger(__name__)

    artifacts_dir = config.paths.artifacts_dir
    qa_path = get_qa_validated_path(artifacts_dir)
    splits_dir = _splits_dir(artifacts_dir)

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

    manifest = {
        "seed": config.project.seed,
        "ratios": config.splits.ratios,
        "stratify_by": config.splits.stratify_by,
        "qa_validated_path": str(qa_path),
        "qa_validated_sha256": _sha256_file(qa_path),
        "qa_validated_count": len(qa_pairs),
        "split_counts": {name: len(splits[name]) for name in SPLIT_NAMES},
        "split_type_counts": {
            name: dict(Counter(qa["type"] for qa in splits[name])) for name in SPLIT_NAMES
        },
    }
    write_json(_manifest_path(artifacts_dir), manifest)
    log.info("Wrote manifest → %s", _manifest_path(artifacts_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--force", action="store_true", help="Overwrite existing splits")
    args = parser.parse_args()
    main(load_config(args.config), force=args.force)
