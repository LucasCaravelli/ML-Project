# Phase 2a → 2b handoff (Me → Quentin, then Max)

This document explains what I shipped on the `oracle-gap` branch (PR #17), the
non-obvious decisions baked into the artifacts, and exactly what Quentin should
consume to start the router track. A short note for Max is at the bottom.

---

## TL;DR for Quentin

You are unblocked. Train on `artifacts/oracle/labels.jsonl` (filtered to
`split=="train"`), select on `split=="val"`, evaluate exactly once on
`split=="test"`. The action space is `{128, 256, 512}` (1024 was retired —
see §3). The class distribution is severely skewed toward 128 (82.1% on test),
so plan for **majority-class default + minority-class override** rather than
expecting a balanced 4-class classifier (see §4). The headline numbers your
router has to beat are in §5.

---

## 1. What I shipped

| Artifact | Path | Purpose |
| --- | --- | --- |
| Stratified splits (60/20/20 by question type, seed=42) | `artifacts/splits/{train,val,test}.jsonl` | Frozen QA partition |
| Splits manifest | `artifacts/splits/manifest.json` | Audit trail (counts, sha256, note) |
| Eval grid (per-question × per-size F1/EM/faithfulness) | `artifacts/oracle/eval_grid.jsonl` | Full grid across **all 4 sizes** {128, 256, 512, 1024} |
| Oracle labels (router target) | `artifacts/oracle/labels.jsonl` | Restricted to active action space {128, 256, 512} |
| Oracle labels (full, for ablation only) | `artifacts/oracle/labels_full.jsonl` | All 4 sizes — do **not** train on this |
| Baselines on test | `results/runs/<ts>_baseline_<size>/metrics.json` | Per-size mean F1 on test |
| Oracle gap reports | `results/oracle_gap.json`, `results/test_summary.json` | Headline numbers |

Code I own/touched:

- `rag_cr/splits.py`, `rag_cr/oracle.py`
- `experiments/make_splits.py`, `experiments/compute_oracle.py`, `experiments/run_baselines.py`
- `experiments/make_figures.py` (figures + tables for the report)
- `configs/base.yaml` (action space `chunking.sizes: [128, 256, 512]`)

---

## 2. Splits are a *frozen artifact* — do not regenerate

The committed splits are **237 / 77 / 84** (train/val/test), totalling **398**.

QA generation originally produced 400 pairs but two `qa_id`s collided due to a
now-fixed offset bug in `qa_gen.py`; the multihop versions were dropped. The
splits were generated *before* that dedup, then preserved across it so that
`eval_grid.jsonl` (which keys on `qa_id`) stays aligned with the splits.

**Do not run `python -m experiments.make_splits --force`.** With the deduped
qa_validated.jsonl as input, `make_splits` deterministically produces a
slightly different partition (237/78/83). That partition is internally valid
but breaks alignment with the existing eval grid for ~1 question, which would
silently invalidate already-computed oracle labels.

The note is permanently recorded in `artifacts/splits/manifest.json` and is now
preserved across regenerations by `make_splits.py` (it carries forward the
prior manifest's `note` unless `--note` is passed explicitly).

If you ever do need to regenerate everything from scratch (you almost
certainly do not), the order is: `make_splits --force` → `build_indices` →
`compute_oracle --splits all` → `run_baselines`. Expect the splits to shift
by a couple of questions.

---

## 3. Action space: 1024 retired

`configs/base.yaml` now has `chunking.sizes: [128, 256, 512]`. The 1024 size was
dropped because at k=2 it produced ~2× the prompt tokens of size 512 with worse
mean F1. The eval grid still contains 1024 rows (so the ablation is reproducible),
but `compute_oracle.py` now writes two label files:

- `labels.jsonl` — restricted to `config.chunking.sizes`. **This is what the
  router trains on.** The router cannot select a size outside the action
  space, so `best_size` must not name a retired size.
- `labels_full.jsonl` — labels over the full grid, written only when the
  active set is a strict subset. For ablation tables only.

The filtering happens in `experiments/compute_oracle.py` immediately before
`label_from_grid` is called.

---

## 4. The skew you need to plan around

Best-size distribution in `labels.jsonl` (restricted action space):

| split | **128** | 256 | 512 |
| ---: | ---: | ---: | ---: |
| test  | **82.1%** | 11.9% | 6.0% |
| train | **83.1%** | 11.0% | 5.9% |
| val   | **88.3%** | 7.8% | 3.9% |

This is *much* more concentrated than the four-size oracle showed — when 1024
was dropped, its share rolled mostly into 128.

Implications for the router track:

1. **The "always pick 128" baseline is the real bar to beat**, not the
   uniform-random baseline. On test, always-128 ≈ F1 0.213 (the size-128
   baseline in `results/test_summary.json`). The oracle ceiling is
   +8.19 pts above the best fixed size.
2. **The README's `_verdict` heuristic in `run_baselines.py` flips to PIVOT**
   because `max_share > 0.60`. This is documented in the README — do not
   "fix" it; it correctly captures that the routing premise is weaker after
   1024 retirement. The defense is: routable headroom is real (+14.63 pts on
   factoid, n=28 in test), it's just concentrated in one question type.
3. **Class-balanced loss / class weighting will matter** for any classifier you
   train. Plain accuracy on val is not a useful selection metric — a
   classifier that always predicts 128 gets ~83% accuracy and recovers 0% of
   the oracle gap. Prefer val *F1 of the resulting RAG system* (per the
   original Phase 2c plan, item 5) or at least balanced accuracy / per-class
   F1 if RAG-F1 selection is too slow.
4. **Type-conditioned routing is the most likely win.** The factoid subset
   has the highest oracle headroom; multihop and synthesis are dominated by
   128. A simple type-aware policy ("default 128, override on factoid") may
   beat a from-scratch 3-class classifier and is worth running as a sanity
   baseline before the full 3×3 grid.

See README §"Distribution: 128 dominates, and the routable headroom is
concentrated in factoid" for the report-level framing.

---

## 5. Headline numbers Quentin must beat

From `results/test_summary.json` and `results/oracle_gap.json` (test split,
n=84):

- Best fixed-size baseline: **size 128, mean F1 ≈ 0.213**
- Oracle ceiling (mean of per-question max F1, restricted to {128,256,512}):
  **≈ 0.295**
- **Oracle gap on test: +8.19 F1 points**
- Always-128 majority-class policy: identical to the size-128 baseline
  (≈ 0.213) by construction

The router's gap-recovery fraction (per README line 17–18) is

```
(router_F1_test − 0.213) / 0.082
```

A router that recovers ≥ ~25% of the gap (i.e. F1 ≥ ~0.234 on test) is a
positive result given the skew; below that, the honest framing is that the
premise is weak under this action space, which is itself a publishable result.

(Numbers above are pulled from the committed `results/`. Re-read the JSONs
before quoting them in a paper — they may have been updated since this
handoff was written.)

---

## 6. Reproducing the oracle / baseline pipeline on the cluster

You will probably not need to re-run this, but if you do:

1. SSH to Bocconi cluster, `cd` into the repo, `git pull`, activate `ragcr`
   conda env.
2. `sbatch experiments/build_indices.slurm` — CPU-only, builds FAISS indices
   per chunk size. **Do not request a GPU for this** (the SLURM file already
   omits one; the bug was fixed in 71043d9).
3. `sbatch experiments/oracle_test.slurm` (test split only) or
   `oracle_full.slurm` (all three splits). Uses vLLM 0.6.3.post1 +
   Qwen/Qwen2.5-7B-Instruct on an A100 MIG slice. Takes ~30 min for all 398
   QAs × 4 sizes.
4. `python -m experiments.run_baselines` — local or cluster, fast.

Cluster gotchas I hit and you will too:

- `git config user.email` / `user.name` are not set globally — set them
  repo-locally on first commit.
- HTTPS git push needs a PAT, not your GitHub password (password auth
  disabled since 2021). SSH also works.
- `tests/test_retrieval.py` fails locally on Windows with a torch DLL error
  (WinError 127). Pre-existing, unrelated, ignore with
  `pytest --ignore=tests/test_retrieval.py`. Other 39 tests pass.
- Re-running the oracle pipeline produces float-precision diffs in
  `test_summary.json` / `oracle_gap.json` (last bit of mantissa). Don't
  commit those — `git checkout --` them.

---

## 7. What changed in PR #17 (one-line each)

- Filter oracle labels to active action space; emit `labels_full.jsonl` for
  ablation. (`experiments/compute_oracle.py`)
- Preserve manifest audit-trail note across `--force` regenerations; add
  `--note` CLI flag; write POSIX paths. (`experiments/make_splits.py`,
  `artifacts/splits/manifest.json`)
- Allowlist `labels_full.jsonl` in `.gitignore`.
- Regenerate `labels.jsonl` against the 3-size action space.
- README: name the 82.1% size-128 concentration; reframe routing as
  majority-class default + factoid override.

---

## 8. Note for Max

You inherit fusion + frontier integration. Your blocking dependency is Quentin's
`run_router.py` writing `results/runs/<ts>_router/metrics.json` in the same
shape as the baselines. Until then you can do all of Phase 2b-i (RRF fusion
on test) standalone — your inputs are `artifacts/splits/test.jsonl` and the
already-built FAISS indices.

Two concrete things to watch for when you arrive:

1. The eval grid contains 1024 rows but the action space does not.
   `FusionSystem` should fuse only across the active sizes
   (`config.chunking.sizes`) unless we explicitly decide otherwise — the
   1024 retrieval costs are real and would inflate fusion cost without buying
   F1.
2. The frontier plot's gap-recovery fraction must use the **restricted**
   oracle ceiling (from `labels.jsonl`), not the full-grid one
   (`labels_full.jsonl`), so the router and fusion are scored against the
   same ceiling they were optimized against.

---

## 9. Router track outcomes (Quentin → Max)

The router track is **complete and closed**. The blocking dependency from §8
is now resolved.

### Results (test split, n=84)

| System | Mean F1 | Gap closure |
|---|---|---|
| Fixed baseline (size 128) | 0.213 | 0.00 |
| Type-aware heuristic | 0.229 | +0.20 |
| **Oracle ceiling** | **0.295** | 1.00 |
| Trained router (MiniLM + LR) | 0.171 | −0.51 |

**Conclusion:** the trained router is a negative result. The only useful
routing signal is coarse question type; the type-aware heuristic (no
training) outperforms both the trained router and the fixed baseline.

### New scripts (all run from `rag-chunk-routing/`)

| Script | Purpose | GPU needed |
|---|---|---|
| `experiments/train_router.py` | CV grid search + two-pass val selection; saves `artifacts/router/best.pkl` | Yes (MiniLM embed) |
| `experiments/run_router.py` | Evaluates best router on test via full RAG pipeline | Yes (vLLM) |
| `experiments/run_type_router.py` | Type-aware sanity baseline; reads eval grid, no inference | No |
| `experiments/make_router_figures.py` | Generates `fig_router_comparison.*` and `fig_router_per_type.*` | No |

Run everything in order with `make router` (depends on `oracle`).

### `metrics.json` schema (your unblocking dependency)

`results/runs/<ts>_router/metrics.json` has this shape — use it for the
frontier plot:

```json
{
  "mean_f1": 0.171,
  "mean_em": 0.107,
  "mean_cost_tokens": 569.1,
  "macro_f1_router_classification": 0.292,
  "balanced_accuracy_router_classification": 0.299,
  "accuracy_router_classification": 0.595,
  "gap_closure_fraction": -0.511,
  "n_test_examples": 84,
  "best_baseline_mean_f1_ref": 0.213,
  "oracle_mean_f1_ref": 0.295
}
```

The type-aware heuristic run (`<ts>_type_router/metrics.json`) has the same
`mean_f1` and `gap_closure_fraction` fields and can be plotted alongside the
trained router on the frontier.

### Committed artifacts

- `results/figures/table_router_results.tex` — LaTeX table (already `\input`'d in the report)
- `results/figures/fig_router_comparison.{pdf,png}` — bar chart
- `results/figures/fig_router_per_type.{pdf,png}` — per-type breakdown
- `tests/test_type_router.py` — 28 tests for the new scripts

### What you do not need to re-run

The report section is written and figures are committed. You only need to
re-run `make router` if you change the pipeline or want fresh predictions.
