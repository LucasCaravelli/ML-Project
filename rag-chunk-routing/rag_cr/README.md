# `rag_cr/` — library code

Importable modules. No CLI entry points (those live in [`../experiments/`](../experiments/)).

## Configuration & infra

- [`config.py`](config.py) — typed dataclass config loaded from YAML; canonical hyperparameter source.
- [`logging.py`](logging.py) — project logger with consistent CLI formatting.
- [`seed.py`](seed.py) — deterministic seeding across Python / NumPy / PyTorch.
- [`io.py`](io.py) — JSONL/JSON helpers and standardized artifact/results path resolution.
- [`types.py`](types.py) — shared `TypedDict`s for chunks, hits, QA pairs, oracle labels, scores.
- [`tokens.py`](tokens.py) — canonical (whitespace) token counting.
- [`utils.py`](utils.py) — shared experiment-script helpers (git hash, oracle-gap, gap-closure).

## Corpus → index → retrieval

- [`corpus.py`](corpus.py) — corpus I/O and metadata accessors.
- [`chunking.py`](chunking.py) — deterministic tokenizer-aware fixed-size chunking.
- [`embedding.py`](embedding.py) — BGE-small dense embedding of chunks and queries.
- [`indexing.py`](indexing.py) — FAISS index build, persist, search per chunk size.
- [`retrieval.py`](retrieval.py) — unified retriever; single-scale and RRF-fused multi-scale.
- [`fusion.py`](fusion.py) — reciprocal-rank fusion across scales.

## QA generation & evaluation

- [`qa_gen.py`](qa_gen.py) — synthetic QA generation via OpenAI, type-stratified.
- [`qa_filter.py`](qa_filter.py) — primary-F1 threshold + LLM-judge filter.
- [`splits.py`](splits.py) — deterministic stratified train/val/test split.
- [`generation.py`](generation.py) — vLLM / ollama answer-generation backends.
- [`metrics.py`](metrics.py) — exact match, token-overlap F1, faithfulness.
- [`oracle.py`](oracle.py) — oracle-label derivation from the eval grid.
- [`pipeline.py`](pipeline.py) — end-to-end retrieve → generate → score orchestration.
- [`systems.py`](systems.py) — system abstractions (FixedSize, Fusion, Oracle, Router) for evaluation.
