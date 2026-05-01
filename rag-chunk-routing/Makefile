PY := python
CONFIG := configs/base.yaml

# macOS: LightGBM requires libomp which Homebrew installs but doesn't add to the default rpath.
# On Linux this is a no-op; on macOS it points the dynamic linker to the Homebrew libomp.
export DYLD_LIBRARY_PATH := /opt/homebrew/opt/libomp/lib:$(DYLD_LIBRARY_PATH)

.PHONY: all indices qa qa-generate qa-filter qa-validate oracle baselines fusion router frontier clean clean-qa test lint format

all: frontier

indices:
	$(PY) experiments/build_indices.py --config $(CONFIG)

qa-generate: indices
	$(PY) experiments/generate_qa.py --config $(CONFIG)

qa-filter:
	$(PY) experiments/filter_qa.py --config $(CONFIG)

qa-validate:
	$(PY) experiments/validate_qa.py --config $(CONFIG)

qa: qa-generate qa-filter qa-validate

oracle: qa
	$(PY) experiments/compute_oracle.py --config $(CONFIG)

baselines: oracle
	$(PY) experiments/run_baselines.py --config $(CONFIG)

fusion: oracle
	$(PY) experiments/run_fusion.py --config $(CONFIG)

router: oracle
	$(PY) experiments/train_router.py --config $(CONFIG)
	$(PY) experiments/run_router.py --config $(CONFIG)

frontier: baselines fusion router
	$(PY) experiments/make_frontier.py --config $(CONFIG)

test:
	pytest

lint:
	ruff check .

format:
	black .
	ruff check --fix .

clean:
	rm -rf artifacts/* results/*

clean-qa:
	rm -rf artifacts/qa
