from __future__ import annotations

import argparse

from rag_cr import Config, get_logger, load_config, set_seed
from rag_cr.pipeline import run


def main(config: Config, qa_path: str | None, limit: int | None) -> None:
    set_seed(config.project.seed)
    log = get_logger(__name__)

    systems = [f"fixed_{size}" for size in config.chunking.sizes] + ["fusion"]

    for system_name in systems:
        log.info("Running system: %s", system_name)
        run_dir = run(system_name, qa_path=qa_path, config=config, limit=limit)
        log.info("  results → %s", run_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--qa-path", default=None, help="Override QA file path")
    parser.add_argument("--limit", type=int, default=None, help="Max QA pairs per system")
    args = parser.parse_args()
    main(load_config(args.config), args.qa_path, args.limit)
