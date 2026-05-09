# `experiments/` — CLI entry points

Each script is a standalone CLI that takes `--config configs/base.yaml`. Library code lives in [`../rag_cr/`](../rag_cr/); orchestration (with dependencies) lives in [`../Makefile`](../Makefile).

## Pipeline order

Run via `make <target>` rather than invoking scripts directly when possible.

1. [`build_indices.py`](build_indices.py) — chunk corpus, embed, build per-size FAISS indices. (`make indices`)
2. [`generate_qa.py`](generate_qa.py) — synthesize raw QA from chunks via OpenAI. (`make qa-generate`)
3. [`filter_qa.py`](filter_qa.py) — primary-F1 + LLM-judge filter into validated/rejected. (`make qa-filter`)
4. [`validate_qa.py`](validate_qa.py) — interactive human review of filtered QA. (`make qa-validate`)
5. [`make_splits.py`](make_splits.py) — stratified train/val/test split of validated QA.
6. [`compute_oracle.py`](compute_oracle.py) — score every (question, chunk-size) cell; emit oracle labels. (`make oracle`)

## Evaluation & analysis

- [`run_baselines.py`](run_baselines.py) — fixed-size baselines + oracle gap. (`make baselines`)
- [`run_fusion.py`](run_fusion.py) — RRF fusion baseline on the test split. (`make fusion`)
- [`train_router.py`](train_router.py) — CV grid search + val re-ranking for the router.
- [`run_router.py`](run_router.py) — evaluate trained router on test split.
- [`run_type_router.py`](run_type_router.py) — type-aware routing sanity baseline (no GPU).
- [`make_frontier.py`](make_frontier.py) — accuracy-cost frontier across all systems. (`make frontier`)
- [`make_figures.py`](make_figures.py) — report tables (LaTeX) and figures (PDF).
- [`make_router_figures.py`](make_router_figures.py) — router-specific figures.
