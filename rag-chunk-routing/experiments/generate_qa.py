from __future__ import annotations

import argparse

from dotenv import load_dotenv

from rag_cr import Config, load_config, set_seed
from rag_cr.io import get_chunks_path, get_qa_raw_path, write_jsonl
from rag_cr.logging import get_logger
from rag_cr.qa_gen import generate_qa

log = get_logger(__name__)


def main(config: Config, limit: int | None) -> None:
    set_seed(config.project.seed)
    load_dotenv()

    chunks_path = get_chunks_path(config.paths.artifacts_dir, config.qa.source_chunk_size)
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunks file missing: {chunks_path}. Run `make indices` first."
        )

    log.info(
        "Generating QA pairs from %s (limit=%s, target_count=%d)",
        chunks_path,
        limit if limit is not None else config.qa.initial_batch_size,
        config.qa.target_count,
    )

    pairs = generate_qa(
        chunks_path=chunks_path,
        qa_cfg=config.qa,
        question_prompt_path=config.prompts.qa_generation,
        answer_prompt_path=config.prompts.qa_answer,
        limit=limit,
    )

    out_path = get_qa_raw_path(config.paths.artifacts_dir)
    write_jsonl(out_path, pairs)
    log.info("Wrote %d raw QA pairs to %s", len(pairs), out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Override qa.initial_batch_size for small test runs.",
    )
    args = parser.parse_args()
    main(load_config(args.config), limit=args.limit)
