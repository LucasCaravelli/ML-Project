from __future__ import annotations

import argparse

from rag_cr import Config, get_logger, load_config, set_seed
from rag_cr.chunking import chunk_corpus
from rag_cr.embedding import embed_texts
from rag_cr.indexing import build_index
from rag_cr.io import get_chunks_path, get_index_path, write_jsonl


def main(config: Config) -> None:
    set_seed(config.project.seed)
    log = get_logger(__name__)

    corpus_path = config.paths.corpus
    artifacts_dir = config.paths.artifacts_dir
    emb = config.embedding

    for size in config.chunking.sizes:
        overlap = config.chunking.overlaps[size]

        log.info("Chunking: size=%d overlap=%d", size, overlap)
        chunks = chunk_corpus(corpus_path, size, overlap)
        chunks_path = get_chunks_path(artifacts_dir, size)
        write_jsonl(chunks_path, list(chunks))
        log.info("  saved %d chunks → %s", len(chunks), chunks_path)

        texts = [c["text"] for c in chunks]
        log.info("  embedding %d chunks with %s", len(texts), emb.model_name)
        embeddings = embed_texts(texts, emb.model_name, emb.device, emb.batch_size, emb.normalize)

        index_path = get_index_path(artifacts_dir, size)
        build_index(embeddings, index_path)
        log.info("  index → %s", index_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    main(load_config(args.config))
