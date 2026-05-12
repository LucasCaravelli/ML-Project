PY := python
CONFIG := configs/base.yaml

# macOS: LightGBM requires libomp which Homebrew installs but doesn't add to the default rpath.
# On Linux this is a no-op; on macOS it points the dynamic linker to the Homebrew libomp.
export DYLD_LIBRARY_PATH := /opt/homebrew/opt/libomp/lib:$(DYLD_LIBRARY_PATH)

.DEFAULT_GOAL := help

.PHONY: help all indices qa qa-generate qa-filter qa-validate oracle baselines fusion router frontier clean clean-qa test lint format

help:  ## Show this help with the list of available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

all: frontier  ## Run the full pipeline (alias for frontier)

indices:  ## Chunk the corpus and build per-size FAISS indices
	$(PY) experiments/build_indices.py --config $(CONFIG)

qa-generate: indices  ## Generate raw QA pairs from corpus chunks (OpenAI API)
	$(PY) experiments/generate_qa.py --config $(CONFIG)

qa-filter:  ## Filter raw QA into validated/rejected via primary-F1 + judge LLM
	$(PY) experiments/filter_qa.py --config $(CONFIG)

qa-validate:  ## Interactive human review of filtered QA pairs
	$(PY) experiments/validate_qa.py --config $(CONFIG)

qa: qa-generate qa-filter qa-validate  ## Run all three QA stages end-to-end

oracle: qa  ## Score the eval grid across every (question, chunk-size) pair
	$(PY) experiments/compute_oracle.py --config $(CONFIG)

baselines: oracle  ## Aggregate per-size baselines and compute the oracle gap
	$(PY) experiments/run_baselines.py --config $(CONFIG)

fusion: oracle  ## Evaluate the RRF fusion baseline on the test split
	$(PY) experiments/run_fusion.py --config $(CONFIG)

router: oracle  ## Train router, evaluate on test, and produce router figures
	$(PY) experiments/train_router.py --config $(CONFIG)
	$(PY) experiments/run_router.py --config $(CONFIG)
	$(PY) experiments/run_type_router.py --config $(CONFIG)
	$(PY) experiments/make_router_figures.py --config $(CONFIG)

frontier: baselines fusion router  ## Build the accuracy-cost frontier figure
	$(PY) experiments/make_frontier.py --config $(CONFIG)

test:  ## Run the test suite
	pytest

lint:  ## Run ruff linter
	ruff check .

format:  ## Auto-format with black and apply ruff fixes
	black .
	ruff check --fix .

clean:  ## Remove all artifacts/ and results/ contents
	rm -rf artifacts/* results/*

clean-qa:  ## Remove only the artifacts/qa/ directory
	rm -rf artifacts/qa
