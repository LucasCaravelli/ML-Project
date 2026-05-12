# `tests/` — pytest suite

Run with `make test` (or `pytest` from the package root).

Test files mirror [`../rag_cr/`](../rag_cr/) 1:1: `test_<module>.py` covers `rag_cr/<module>.py`. Add a new test file alongside any new `rag_cr` module.

Shared fixtures live in [`conftest.py`](conftest.py).

## Coverage notes

- No tests for thin wrappers (`logging`, `seed`, `tokens`, `types`) or for backends that require external services (`generation`, `embedding`, `indexing`, `pipeline`). `qa_gen` and `qa_filter` are tested with the LLM call monkeypatched.
- Experiment scripts in [`../experiments/`](../experiments/) are not unit-tested; they are exercised end-to-end via the `make` pipeline.

## Markers

- `integration`: requires built artifacts on disk (e.g. [`test_retrieval.py`](test_retrieval.py) needs `artifacts/indices/` from `make indices`). Run `pytest -m "not integration"` to skip.
- `slow`: loads large models. Run `pytest -m "not slow"` to skip.
