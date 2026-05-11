# RAG Chunk-Size Routing

Can a cheap, query-only classifier pick the right chunk size for RAG retrieval
and recover the per-query oracle gap without the cost of multi-scale fusion?
This repo tests that question on a Friedreich Ataxia medical QA corpus
(398 validated question-answer pairs, action space {128, 256, 512} tokens,
Qwen2.5-7B generator on an A100).

**Main result:** the trained router (MiniLM + logistic regression) scores
*below* the best fixed-size baseline (F1 0.171 vs 0.213, gap closure −0.51).
A parameter-free type-aware heuristic achieves F1 0.229 (+0.20 gap closure).
RRF multi-scale fusion achieves F1 0.223 at ~44 % higher retrieval cost.

---

## Results

All numbers are on the **test split (n = 84)**, action space {128, 256, 512}.

| System | F1 | Gap closure | Mean tokens |
| --- | ---: | ---: | ---: |
| Fixed (size 128) | 0.213 | 0.00 | ~570 |
| Fixed (size 256) | 0.170 | −0.52 | ~570 |
| Fixed (size 512) | 0.170 | −0.53 | ~570 |
| RRF Fusion | 0.223 | +0.13 | ~820 |
| Type-aware heuristic | 0.229 | +0.20 | ~570 |
| Router (MiniLM + LR) | 0.171 | −0.51 | ~570 |
| **Oracle ceiling** | **0.295** | 1.00 | ~570 |

Gap closure = (system F1 − best baseline F1) / (oracle F1 − best baseline F1).
Oracle gap = **+8.19 F1 points**.

Figures and LaTeX tables: `rag-chunk-routing/results/figures/`.
Per-run outputs: `rag-chunk-routing/results/runs/<timestamp>_<system>/metrics.json`.

---

## Repo layout

```
ML-Project/
├── neurips_2026.tex / .sty     # Paper draft (NeurIPS 2026 template)
├── REPORT_GUIDE.md             # Guide for report writers — start here
├── LitReview.tex               # Literature review
│
└── rag-chunk-routing/
    ├── configs/                # YAML hyperparameter files
    ├── data/                   # Raw corpus (read-only)
    ├── artifacts/              # Derived data — rebuildable from scripts
    │   ├── chunks/             # Chunked corpus (.jsonl per size)
    │   ├── indices/            # FAISS dense-retrieval indices
    │   ├── qa/                 # Validated QA pairs
    │   ├── splits/             # Train / val / test splits
    │   ├── oracle/             # Oracle labels and eval grid
    │   ├── baselines/          # Baseline metrics and oracle gap JSON
    │   └── router/             # Trained router pickle and CV results
    ├── experiments/            # CLI scripts (one per pipeline stage)
    ├── rag_cr/                 # Reusable Python library
    ├── results/                # Timestamped run outputs and figures
    ├── prompts/                # LLM prompt templates
    ├── slurm/                  # HPC job scripts
    └── tests/                  # pytest suite (~167 tests)
```

---

## Setup

```bash
cd rag-chunk-routing
pip install -e ".[dev]"           # local development
pip install -e ".[dev,cluster]"   # cluster (adds vllm)
cp .env.example .env              # fill in OPENAI_API_KEY (QA generation only)
```

Exact dependency pins for reproducing committed results:

```bash
pip install -r rag-chunk-routing/requirements.lock
```

---

## Running the pipeline

Run from `rag-chunk-routing/` via `make`. Pre-built artifacts are committed,
so you can start from `make baselines` without re-running the GPU stages.

| Target | What it does | GPU |
| --- | --- | --- |
| `make indices` | Chunk corpus, embed, build FAISS indices | No |
| `make oracle` | Score every (question, chunk-size) pair | Yes |
| `make baselines` | Fixed-size metrics + oracle gap | No |
| `make fusion` | RRF fusion evaluation on test split | Yes |
| `make router` | Train + evaluate router, generate figures | Yes* |
| `make figures` | Report tables (LaTeX) and figures (PDF) | No |
| `make frontier` | Accuracy–cost frontier plot | No |
| `make test` | Full pytest suite | No |
| `make all` | Full rebuild from raw corpus | Yes |

\* `run_type_router.py` (included in `make router`) needs no GPU.

On the cluster, submit SLURM scripts instead of running `make oracle` /
`make fusion` directly:

```bash
sbatch slurm/build_indices.slurm    # CPU — run first
sbatch slurm/oracle_test.slurm      # test-split oracle (A100)
sbatch slurm/oracle_full.slurm      # train/val oracle  (A100)
sbatch slurm/fusion.slurm           # fusion eval       (A100)
```

---

## Configs

| File | Backend | Purpose |
| --- | --- | --- |
| `configs/base.yaml` | ollama | Local development |
| `configs/cluster.yaml` | vllm + Qwen2.5-7B | Cluster (A100) |
| `configs/eval_dry_run.yaml` | extractive (no GPU) | CI / offline tests |

Pass `--config <path>` to any experiment script. Every hyperparameter lives
in the config; there are no magic numbers in Python.

---

## Key artifacts

| Path | Contents |
| --- | --- |
| `artifacts/oracle/eval_grid.jsonl` | 398 × 4 scored (question, chunk-size) grid — primary data source |
| `artifacts/baselines/oracle_gap.json` | Oracle gap summary and per-type breakdown |
| `artifacts/router/best.pkl` | Trained router (MiniLM + LR), fitted on full training set |
| `artifacts/router/cv_results.csv` | Full 3×3 CV grid results |
| `results/figures/frontier.{pdf,png}` | Accuracy–cost frontier (headline figure) |
| `results/figures/table_*.tex` | LaTeX tables included directly in the paper |

---

## Tests

```bash
make test                        # full suite (~167 tests)
pytest -m "not integration"      # skip tests that need built artifacts
pytest -m "not slow"             # skip tests that load MiniLM
```

CI (`.github/workflows/ci.yml`) runs the torch-free subset on every push
to `main`: `test_metrics`, `test_oracle`, `test_config`, `test_splits`,
`test_fusion`, `test_io`, `test_utils`, `test_corpus`.
