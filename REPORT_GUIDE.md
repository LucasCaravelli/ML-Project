# RAG Chunk-Routing — Project & Report Writer Guide

> General Project Layout and Advice for Report Writers

---

## 1. What We Are Building

We test a single question: *can a cheap, query-only classifier pick the right
chunk size for RAG retrieval and recover most of the "oracle gap" without paying
the cost of full retrieval fusion?*

**The oracle gap** is the difference between the best any fixed chunk size achieves
and what a perfect router would achieve if it always picked the best size per
question. On our test set that gap is **8.19 F1 points**:

- Best fixed-size baseline (128-token chunks): F1 = **0.2128**
- Oracle ceiling: F1 = **0.2947**
- Gap: **8.19 F1 points**

We compare four system families:

| System | Description |
|---|---|
| Fixed-128 / 256 / 512 | Always retrieve from one chunk-size index |
| RRF Fusion | Retrieve from all three indices, merge with Reciprocal Rank Fusion |
| Oracle ceiling | Always picks the per-question best size (unreachable upper bound) |
| Router | Learned classifier predicts best chunk size from query features alone |

The corpus is medical literature on **Friedreich Ataxia** (~1.4 MB). The QA set
has **398 validated question-answer pairs** across three types: factoid, multihop,
and synthesis.

---

## 2. Repository Layout

```
ML-Project/
├── LitReview.tex              # Literature review
├── neurips_2026.tex           # NeurIPS 2026 paper template
├── neurips_2026.sty           # NeurIPS style file
├── README.md                  # Original team working document
├── REPORT_GUIDE.md            # ← this file
│
└── rag-chunk-routing/         # Everything lives here
    ├── configs/               # YAML hyperparameter files
    ├── data/                  # Raw corpus (never modified)
    ├── artifacts/             # Derived data (rebuildable from scripts)
    │   ├── chunks/            # Chunked corpus (JSONL per size)
    │   ├── indices/           # FAISS retrieval indices
    │   ├── qa/                # Validated QA pairs
    │   ├── splits/            # Train/val/test split indices
    │   ├── oracle/            # Oracle labels and evaluation grid
    │   └── baselines/         # ★ Baseline metrics — primary figure data ★
    ├── experiments/           # CLI entry-point scripts (run these)
    ├── rag_cr/                # Reusable Python library
    ├── results/               # Timestamped run outputs
    │   └── 20260508_000000_fusion/   # ★ Latest fusion run ★
    │   └── 20260508_000001_router/   # ★ Latest router run ★
    ├── prompts/               # LLM prompt templates
    ├── slurm/                 # HPC job scripts
    └── tests/                 # Unit tests
```

### Three-tier data philosophy

| Tier | Location | Rule |
|---|---|---|
| **Raw** | `data/` | Never touch. Read-only forever. |
| **Artifacts** | `artifacts/` | Rebuildable by running the pipeline scripts. |
| **Results** | `results/` | Timestamped, append-only — never overwrite a past run. |

---

## 3. Artifacts — Detailed Map

### `artifacts/chunks/`

Chunked corpus in JSONL format. Each line: `{chunk_id, size, start_char, end_char, text}`.

| File | Chunks |
|---|---|
| `128.jsonl` | 2,911 |
| `256.jsonl` | fewer (larger chunks) |
| `512.jsonl` | fewer still |
| `1024.jsonl` | fewest (dropped from canonical experiments) |

### `artifacts/indices/`

FAISS dense-retrieval indices built with BAAI/BGE-small-en-v1.5 embeddings on CPU.
One `.faiss` file per chunk size.

### `artifacts/qa/`

| File | Contents |
|---|---|
| `qa_validated.jsonl` | 398 human-reviewed QA pairs: `{qa_id, question, answer, source_chunk_id, type, validated}` |
| `qa_rejected.jsonl` | Filtered pairs with reject reasons |

Type breakdown by split:

| Type | Train | Val | Test |
|---|---|---|---|
| factoid | 80 | 25 | 28 |
| multihop | 78 | 26 | 28 |
| synthesis | 79 | 26 | 28 |
| **Total** | **237** | **77** | **84** |

### `artifacts/splits/`

Stratified split (seed=42). `manifest.json` contains audit metadata.
`train.jsonl`, `val.jsonl`, `test.jsonl` contain split indices.

### `artifacts/oracle/`

| File | Contents |
|---|---|
| `eval_grid.jsonl` | Full 398×4 grid: every (question, chunk_size) pair scored with `em, f1, faithfulness, cost_tokens` |
| `labels.jsonl` | Per-question oracle label: `{qa_id, best_size, scores_by_size}` for the canonical 3-size action space |
| `labels_full.jsonl` | Same but including size 1024 (for the ablation) |

### `artifacts/baselines/` — Primary data source for figures

Six JSON files summarising system performance on the **test split**.

#### `test_summary.json` — Fixed-size baselines (canonical 3 sizes)

```json
{
  "128": {"n": 84, "f1": 0.2128, "em": 0.1429, "faithfulness": 0.3626},
  "256": {"n": 84, "f1": 0.1947, "em": 0.1190, "faithfulness": 0.3451},
  "512": {"n": 84, "f1": 0.1830, "em": 0.1190, "faithfulness": 0.3396}
}
```

#### `test_summary_full.json` — Fixed-size baselines including 1024

Size 1024: F1 = 0.1573. Confirms that adding 1024 is strictly worse than any of
the canonical three sizes.

#### `oracle_gap.json` — Oracle ceiling analysis (canonical 3 sizes)

```json
{
  "action_space": [128, 256, 512],
  "oracle_f1_mean": 0.2947,
  "best_baseline_f1": 0.2128,
  "gap_f1_points": 8.19,
  "per_type": {
    "factoid":  {"oracle_f1": ..., "best_baseline_f1": ..., "gap": 14.63},
    "multihop": {"..."},
    "synthesis":{"..."}
  }
}
```

**Factoid questions have the largest gap (~14.63 pts).**

#### `oracle_gap_full.json` — Oracle gap with all 4 sizes

Gap with 1024 included is 10.61 pts (larger than the canonical 8.19 pts because
oracle can sometimes exploit 1024). Used to justify keeping the canonical 3-size
action space for the main experiments.

#### `size_distribution.json` — Oracle-best size distribution (canonical)

Which chunk size is "best" per question, by split:

| Size | Train | Val | Test |
|---|---|---|---|
| 128 | 83.1% | 88.3% | 82.1% |
| 256 | 11.0% | 7.8% | 11.9% |
| 512 | 5.9% | 3.9% | 6.0% |

**Key insight:** 128 wins 82% of the time. The learning problem is heavily
class-imbalanced, which is why a naive always-128 classifier achieves decent
accuracy but fails on F1 gap recovery.

#### `size_distribution_full.json` — Same with 1024

1024 captures near-zero oracle selections — further justifying its exclusion.

---

## 4. Results — Run Outputs

All runs live under `rag-chunk-routing/results/` as timestamp-prefixed folders.

### `results/SUMMARY.md`

Auto-generated leaderboard comparing all systems evaluated so far. **Always check
this first** before opening individual run folders.

### Key runs from 2026-05-08

#### `results/20260508_000000_fusion/metrics.json`

RRF Fusion on the test split:

| Metric | Value |
|---|---|
| F1 | **0.2233** |
| EM | 0.1429 |
| Faithfulness | 0.3669 |
| Total cost tokens | 68,815 |
| Mean cost per query | ~819 tokens |

Fusion recovers **~12.8% of the oracle gap** at the cost of querying all three
indices simultaneously.

#### `results/20260508_000001_router/metrics.json`

Learned Router on the test split:

| Metric | Value |
|---|---|
| F1 | **0.171** |
| EM | 0.107 |
| Router macro F1 (classification) | 0.292 |
| Mean cost per query | 569.1 tokens |
| Gap closure fraction | **−0.511** |

The router currently **underperforms** the best fixed baseline by a large margin
(−51% gap recovery). This is the central negative result of the paper.

### Historical runs

| Timestamp prefix | Type | Notes |
|---|---|---|
| `20260507_164140` | fusion | Previous fusion run |
| `20260507_163556` | fusion | Earlier fusion attempt |
| `20260506_*` | fusion | Multiple earlier fusion iterations |
| `20260430_135557` | oracle | Oracle ceiling evaluation |
| `20260430_135552` | fixed_512 | Fixed-512 baseline |
| `20260428_202144` | fixed_512 | Earlier fixed-512 run |
| `20260421_115611` | fixed_256 | Fixed-256 baseline |

---

## 5. Code Structure

### `configs/`

| File | Purpose |
|---|---|
| `base.yaml` | Master config: chunk sizes `[128,256,512]`, embedder (BGE-small), generation model (Qwen2.5-7B via Ollama), router feature/classifier grid |
| `cluster.yaml` | HPC variant (vLLM backend) |
| `eval_dry_run.yaml` | Quick smoke-test |

All hyperparameters live here — never hardcode values in scripts.

### `experiments/` — Pipeline entry points (run in order)

```
1.  build_indices.py       Chunk corpus → embed → build FAISS indices
2.  generate_qa.py         Synthetic QA via OpenAI
3.  filter_qa.py           Primary-F1 + LLM-judge filtering
4.  validate_qa.py         Interactive human review
5.  make_splits.py         Stratified train/val/test split
6.  compute_oracle.py      Score all (question, size) pairs
7.  run_baselines.py       Fixed-size metrics + oracle gap → artifacts/baselines/
8.  run_fusion.py          RRF Fusion evaluation → results/
9.  train_router.py        CV grid search + val re-ranking
10. run_router.py          Router evaluation → results/
11. make_frontier.py       Accuracy-cost frontier plot
12. make_figures.py        Report tables (LaTeX) and figures
13. make_router_figures.py Router-specific visualizations
```

### `rag_cr/` — Reusable library

| Module | Role |
|---|---|
| `chunking.py` | Tokenizer-aware fixed-size chunking |
| `embedding.py` | BGE-small dense embeddings |
| `indexing.py` | FAISS build/persist/search |
| `retrieval.py` | Single-scale and RRF Fusion retrieval |
| `metrics.py` | EM, F1, faithfulness scoring |
| `oracle.py` | Oracle label derivation |
| `systems.py` | System abstractions (FixedSize, Fusion, Oracle, Router) |
| `router/features.py` | TF-IDF, MiniLM, and handcrafted query feature extractors |
| `router/models.py` | LogReg, LinearSVM, LightGBM classifier wrappers |
| `router/train.py` | 5-fold CV grid search + val re-ranking |

---

## 6. Guide for Report Writers

This section is for team members creating **figures and tables** for the NeurIPS
2026 submission. **You do not need to re-run any experiments.** All data is
already in `artifacts/baselines/` and `results/20260508_*/`.

---

### Data cheat sheet

| What you need | File | Key |
|---|---|---|
| Fixed-size F1 / EM / faithfulness | `artifacts/baselines/test_summary.json` | `"128"`, `"256"`, `"512"` |
| Oracle ceiling | `artifacts/baselines/oracle_gap.json` | `oracle_f1_mean`, `gap_f1_points` |
| Per-type oracle gap | `artifacts/baselines/oracle_gap.json` | `per_type` |
| Oracle-best size distribution | `artifacts/baselines/size_distribution.json` | `test`, `train`, `val` |
| 4-size ablation baselines | `artifacts/baselines/test_summary_full.json` | adds `"1024"` |
| 4-size ablation oracle gap | `artifacts/baselines/oracle_gap_full.json` | — |
| Fusion results | `results/20260508_000000_fusion/metrics.json` | `f1`, `em`, `faithfulness`, `cost_tokens_total` |
| Router results | `results/20260508_000001_router/metrics.json` | `f1`, `em`, `gap_closure_fraction`, `mean_cost_tokens` |
| Per-question predictions + cost | `artifacts/oracle/eval_grid.jsonl` | `f1`, `cost_tokens`, `type`, `chunk_size` |

---

### Figures to produce

#### Figure 1 — Main results bar chart

**Goal:** Show all five systems vs. oracle ceiling on F1.

- **X-axis:** Systems — Fixed-128, Fixed-256, Fixed-512, Fusion, Router
- **Y-axis:** F1 on test split
- **Add** a horizontal dashed line at Oracle F1 = 0.2947, labelled "Oracle ceiling"
- Optionally add error bars from `eval_grid.jsonl` (bootstrap or per-type std)

Numbers to plot:

| System | F1 | Source |
|---|---|---|
| Fixed-128 | 0.2128 | `test_summary.json` |
| Fixed-256 | 0.1947 | `test_summary.json` |
| Fixed-512 | 0.1830 | `test_summary.json` |
| Fusion | 0.2233 | `results/20260508_000000_fusion/metrics.json` |
| Router | 0.171 | `results/20260508_000001_router/metrics.json` |
| Oracle | 0.2947 | `oracle_gap.json` |

---

#### Figure 2 — Per-type oracle gap breakdown

**Goal:** Show where the gap is largest (factoid >> multihop >> synthesis).

- **Type:** Horizontal bar chart, one row per question type
- **Bars:** oracle F1 (full) with best-baseline F1 marked inside (stacked or grouped)
- **Data source:** `oracle_gap.json` → `per_type` field

**Key message for caption:** Factoid questions carry the largest gap (~14.6 pts),
explaining why the router's failure on factoid is the dominant contributor to its
overall underperformance.

---

#### Figure 3 — Oracle-best chunk size distribution

**Goal:** Show class imbalance (why the router defaults to predicting 128).

- **Type:** Stacked horizontal bar chart, one bar per split (train / val / test)
- **Segments:** 128 (blue), 256 (orange), 512 (green)
- **Data source:** `size_distribution.json`

**Key message for caption:** 128-token chunks are oracle-best for ~82% of test
questions — the router must overcome severe class imbalance to improve over
always-predict-128.

---

#### Figure 4 — Accuracy vs. retrieval cost frontier

**Goal:** Position each system on a cost-effectiveness plane.

- **X-axis:** Mean retrieval cost per query (tokens)
- **Y-axis:** F1 on test split
- **Each system is one point.** Draw a Pareto frontier line connecting non-dominated
  points.

Approximate values (compute exact per-query means from `eval_grid.jsonl`
`cost_tokens` field if needed):

| System | Approx. mean cost tokens | F1 |
|---|---|---|
| Fixed-128 | ~570 | 0.2128 |
| Fixed-256 | ~570 | 0.1947 |
| Fixed-512 | ~570 | 0.1830 |
| Fusion | ~820 | 0.2233 |
| Router | 569.1 | 0.171 |
| Oracle | ~570 | 0.2947 |

**Key message for caption:** The router sits at single-index cost but below even
Fixed-128 quality — Fusion achieves better F1 at only moderate extra cost.

---

#### Figure 5 — 4-size ablation (supplementary)

**Goal:** Justify dropping size 1024 from the canonical action space.

Same bar chart structure as Figure 1 but add Fixed-1024 (F1 = 0.1573,
`test_summary_full.json`). The oracle gap also widens (10.61 pts → 8.19 pts) when
1024 is dropped, which actually shrinks the gap — explain this in the caption using
`oracle_gap_full.json`.

---

#### Table 1 — Main results (LaTeX)

Produce a LaTeX `booktabs` table:

| System | F1 | EM | Faithfulness | Cost (tokens) | Gap Recovery |
|---|---|---|---|---|---|
| Fixed-128 | 0.2128 | 0.1429 | 0.3626 | ~570 | 0% (baseline) |
| Fixed-256 | 0.1947 | 0.1190 | 0.3451 | ~570 | −21.8% |
| Fixed-512 | 0.1830 | 0.1190 | 0.3396 | ~570 | −36.1% |
| RRF Fusion | 0.2233 | 0.1429 | 0.3669 | ~820 | +12.8% |
| Router | 0.171 | 0.107 | — | 569.1 | −51.1% |
| Oracle | 0.2947 | — | — | ~570 | 100% |

Gap Recovery formula:
```
gap_recovery = (system_F1 − best_baseline_F1) / (oracle_F1 − best_baseline_F1)
             = (system_F1 − 0.2128) / (0.2947 − 0.2128)
```

Pre-computed values:
- Fusion: (0.2233 − 0.2128) / 0.0819 = **+12.8%**
- Router: (0.171 − 0.2128) / 0.0819 = **−51.1%**

---

### Existing figure scripts

`experiments/make_figures.py` and `experiments/make_router_figures.py` already
exist. **Check these first** — they may already read the right files and only need
minor updates for the latest 20260508 run paths. Only write new plotting code if
these scripts are missing a specific figure you need.

---

### Plotting conventions

- Use **matplotlib** (or seaborn on top of it).
- Target NeurIPS column width = **3.25 in** (single column) or **6.75 in** (full).
- Use a **colourblind-safe palette** (`seaborn colorblind` or ColorBrewer Set2).
- Export as **PDF** for the LaTeX submission; **PNG at 300 dpi** for slides.
- Set `plt.rcParams["font.size"] = 9` to match NeurIPS body text.
- All data is deterministic — figures must be reproducible with a fixed script and
  no random seed required.

---

## 7. Current Status

| Phase | Status | Description |
|---|---|---|
| Phase 1 | ✅ Done | Infrastructure: chunking, indexing, QA generation, validated splits |
| Phase 2 | ✅ Done | Core experiments: all systems have test-set numbers |
| Phase 3 | 🔄 In progress | Freeze, ablations, paper figures |

**The router underperforms.** The paper frames this as a *negative result*: a
simple query-only classifier cannot reliably select chunk size, and the difficulty
stems from (a) severe class imbalance toward 128, (b) limited training data (237
examples), and (c) the absence of any retrieval signal in the router's features.

---

## 8. Reproduction Checklist (for developers, not report writers)

To reproduce results from scratch:

```bash
cd rag-chunk-routing

# Build everything up to oracle labels
python experiments/build_indices.py   --config configs/base.yaml
python experiments/compute_oracle.py  --config configs/base.yaml

# Run evaluations
python experiments/run_baselines.py   --config configs/base.yaml
python experiments/run_fusion.py      --config configs/base.yaml
python experiments/train_router.py    --config configs/base.yaml
python experiments/run_router.py      --config configs/base.yaml

# Generate figures
python experiments/make_figures.py
python experiments/make_router_figures.py
```

`qa_validated.jsonl` and `eval_grid.jsonl` are already generated. Regenerating
them requires OpenAI API calls and Ollama inference — expensive and unnecessary
unless you suspect data corruption.

---

*Last updated: 2026-05-09*
