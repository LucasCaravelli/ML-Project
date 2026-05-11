# RAG Chunk-Size Routing: Team Working Document

This is the working document for the six of us building the project. It is
not a public README. It explains what we are building, why the repo is shaped
the way it is, and how we are going to work across the three phases leading up
to the experiment freeze.
(Test)
## What we are actually doing

We are testing whether a cheap, query-only classifier can pick the right
chunk size for a RAG pipeline well enough to recover a meaningful fraction of
the per-query oracle gap, without paying the retrieval cost that multi-scale
fusion incurs. That is the one question. Every design choice in this repo
serves it. If something does not help us measure the gap or the gap-closure,
it does not belong in the project.

The headline number in the report is the fraction of the oracle gap recovered
per unit retrieval cost. Everything else is supporting evidence.

## Why the repo looks the way it does

Three things about the structure are worth internalizing before anyone starts
writing code, because they will save us from avoidable arguments later.

**Three data tiers, strictly separated.** `data/` holds the raw corpus
(`data/corpus.txt`) and is never written to. `artifacts/` holds everything
derived from the corpus plus code (chunks, indices, QA pairs, oracle labels,
trained router) and can be wiped and rebuilt at any time. `results/` holds
timestamped run outputs and is never overwritten in place. If your change breaks
chunking, you delete artifacts and rebuild; the QA we manually validated and the
results we have already committed survive untouched. See "Data, pre-built
artifacts, and prompts" for the full listing of every committed file in
`artifacts/`.

**Package code in `rag_cr/`, orchestration in `experiments/`.** The package
contains reusable logic behind frozen interfaces. The experiment scripts are
thin: they parse arguments, call package functions, write outputs. If you
find yourself writing real logic inside an `experiments/` script, stop and
move it into the package.

**Every hyperparameter lives in `configs/base.yaml`.** No magic numbers in
Python. This is non-negotiable: it is what lets us rerun an ablation by
editing one file.

## Ground rules

- Interfaces declared in the package are frozen at kickoff. Any change needs
  a consensus, to prevent everything crashing last minute.
- Black and ruff as pre-commit hooks, type hints on public functions, no
  import-time side effects anywhere.
- Every experiment script accepts `--config` and writes to
  `results/runs/<timestamp>_<system>/` with a `meta.json` recording the git
  commit, the config used, and the seed.
- No LLM-as-judge at evaluation time. All scoring is deterministic (EM, F1,
  token-overlap faithfulness). Using an LLM for QA generation is fine and
  expected; using one to score our own system undermines the rigor argument.
- Notebooks are for exploration only. If a function is longer than a handful
  of lines, it moves into `rag_cr/`.

## Week 1: Infrastructure

Goal of this phase: a single command takes one question, retrieves at any
chunk size, generates an answer, and scores it end-to-end.

Work to complete:

- Chunking of the corpus at all four configured sizes.
- FAISS index construction and persistence for each chunk size.
- The unified retrieval interface, working against a hardcoded query.
- Generator wrapper and the answer prompt.
- Exact-match and token-level F1 scorers.
- A generator smoke test on the hardware we actually plan to use. If the 7B
  model is too slow, we fall back to a 3B model or to Ollama on a GPU. We
  decide this in Phase 1, not Phase 3.
- A first batch of synthetic QA pairs (on the order of 150, not 400) and the
  manual validation tool.
- Manual review of the first batch, targeting around 120 clean pairs. We
  would rather have 120 excellent pairs than 400 mediocre ones. The review
  is scheduled as a block of time, not left as something to do "whenever".
- Orchestration skeleton so that one command executes the end-to-end path.

Parallelizable: chunking and indexing, generator wrapper and metrics, QA
generation and validation, and orchestration scaffolding are all independent
streams in this phase. The retrieval interface blocks only the orchestration
scaffolding's final wiring, not its skeleton.

Phase 1 ends when the end-to-end smoke test passes. If it does not pass, we
stop and fix it before starting Phase 2. Moving on with a broken foundation
is how projects die in the last week.

## Phase 2: Core experiments

Goal of this phase: every system in the comparison has numbers on the
frontier plot.

Work to complete:

- Scaling the QA set to its full size and completing validation.
- Oracle labeling: for every validated question, run all four chunk sizes,
  pick the best by F1, tie-break by EM then by smaller chunk size.
- A first look at the distribution of oracle-best chunk sizes immediately
  after labels exist. If the distribution is close to uniform, the premise
  of the project is weaker than we assumed and the routing target needs to
  be reconsidered before more effort goes into the router. Better to
  discover this in Phase 2 than Phase 3.
- Evaluation of the four fixed-size baselines across the full QA set, with
  all outputs cached.
- Evaluation of the RRF multi-scale fusion baseline across the full QA set.
- Implementation of the candidate feature extractors (TF-IDF, MiniLM
  embeddings, handcrafted query features) behind a uniform interface.
- Implementation of the candidate classifier wrappers (logistic regression,
  linear SVM, LightGBM) behind a uniform interface.
- Cross-validated training across the full feature-by-classifier grid, with
  probability calibration, producing a CV results table and a selected best
  configuration.
- Evaluation of the selected router across the full QA set.
- First draft of the accuracy-cost frontier plot with all six systems.

Parallelizable: once oracle labels exist, the fixed-size baselines, the
fusion baseline, and router training are three independent streams. QA
scaling can overlap with oracle design work. The frontier plot can be
drafted against partial results and refreshed as each system lands.

Phase 2 ends when all six systems appear as points on the frontier plot.

## Action space: why we dropped chunk size 1024

This section is written so the report writers can lift the numbers and the
argument directly. Every claim below is backed by a JSON file in
`rag-chunk-routing/artifacts/baselines/`. Numbers are the ones produced on
2026-05-01 from `eval_grid.jsonl` (398 unique QA pairs, 4 chunk sizes) on
the test split (n = 84).

### The decision

The router's action space is **{128, 256, 512}**. Chunk size **1024** was
retired from the action space — it is still chunked, indexed, and present
in the eval grid (so the ablation below remains reproducible), but the
router will never select it and the headline baselines do not consider it.
This is encoded in `chunking.sizes` in both `configs/base.yaml` and
`configs/cluster.yaml`. The 1024 entry in `chunking.overlaps` is
intentionally retained so the existing 1024-size FAISS index and chunk
file stay readable.

### Why: 1024 is the dominated arm

Per-size baseline F1 on the test split, evaluated across all four sizes
(`artifacts/baselines/test_summary_full.json`):

| chunk size | F1         | EM         | faithfulness |
| ---------: | ---------: | ---------: | -----------: |
|        128 | 0.2128     | 0.1429     | 0.3626       |
|        256 | 0.1701     | 0.0714     | 0.3635       |
|        512 | 0.1695     | 0.0833     | 0.3140       |
|   **1024** | **0.1573** | **0.0714** | **0.3092**   |

1024 is the worst single-size baseline on every metric we care about (F1,
EM, faithfulness). Including it in the action space asks the router to
sometimes choose a size that, before the router even sees the question, is
known to be worse on average than the next-worst alternative.

### Why: 1024 is also the rarest oracle winner

Oracle-best-size distribution per split, full action space
(`artifacts/baselines/size_distribution_full.json`):

| split |  128  |  256  |  512  | **1024** |
| ----: | ----: | ----: | ----: | -------: |
|  test | 76.2% | 11.9% |  6.0% | **6.0%** |
| train | 79.7% |  9.7% |  5.5% | **5.1%** |
|   val | 81.8% |  7.8% |  1.3% | **9.1%** |

1024 wins the per-question oracle on roughly 5–9 % of questions across the
splits. A 4-way classifier with one class at this prior is a poor target:
the router gets very little training signal for a class that is, by the
table above, the worst single-size baseline anyway. We avoid that by
narrowing the action space.

Under the restricted action space, the 1024-winners roll almost entirely
into 128 (`artifacts/baselines/size_distribution.json`):

| split |  **128**  |  256  |  512  |
| ----: | --------: | ----: | ----: |
|  test | **82.1%** | 11.9% |  6.0% |
| train | **83.1%** | 11.0% |  5.9% |
|   val | **88.3%** |  7.8% |  3.9% |

This concentration is the dominant fact about the routing problem on this
corpus and is treated separately under "Distribution" below.

### Cost: the headline gap shrinks but stays well above the healthy threshold

Oracle gap on the test split, before and after dropping 1024 (overall and
per question type):

|           | full action space | restricted action space |  sacrifice |
| --------- | ----------------: | ----------------------: | ---------: |
| overall   |        +10.61 pts |               +8.19 pts |  -2.42 pts |
| factoid   |        +15.82 pts |              +14.63 pts |  -1.19 pts |
| multihop  |         +5.82 pts |               +3.02 pts |  -2.80 pts |
| synthesis |         +5.32 pts |               +2.05 pts |  -3.27 pts |

Sources: `artifacts/baselines/oracle_gap.json` (canonical, 3-size) and
`artifacts/baselines/oracle_gap_full.json` (ablation, 4-size).

The +8.19-point gap is well above our 3-point "healthy gap" threshold
(declared in `experiments/run_baselines.py::_verdict`), so the project
premise — that per-question routing beats any fixed size — holds under
the restricted action space. The cost is honest: factoid is essentially
unaffected, but multihop and synthesis lose roughly half of their oracle
headroom. The report should disclose this rather than hide it.

### Distribution: 128 dominates, and the routable headroom is concentrated in factoid

Under the restricted action space, **128 wins the per-question oracle on
82.1 % of test questions**, with 256 at 11.9 % and 512 at 6.0 %. This
trips the second branch of `_verdict` (`max_share > 0.60` ⇒ PIVOT)
even though the gap clears the 3-point healthy threshold; running
`run_baselines.py` after this commit prints the verdict line "PIVOT --
gap too small or one chunk size dominates." Both signals are real, and
the report has to engage with them rather than pick the convenient one.

The honest reading is that **the routing problem on this corpus is not a
balanced 3-way classification**. It is a majority-class default with
selective override, and the override is concentrated in one question
type:

- **Factoid (n = 28 on test)**: best fixed-size F1 = 0.486 (size 128) →
  oracle F1 = 0.633, **+14.63 F1 points** of routable headroom. The
  factoid winner distribution across {128, 256, 512} is genuinely mixed
  (no single size wins more than ~50 %, see
  `artifacts/oracle/labels.jsonl` filtered by `type=factoid`), so this is
  where a learned router can plausibly capture real lift.
- **Multihop (n = 28)**: gap +3.02 pts on top of a low absolute F1
  (0.125 best-baseline). Small absolute headroom; multihop is bounded by
  retrieval / generation quality, not by chunking choice.
- **Synthesis (n = 28)**: gap +2.05 pts on F1 = 0.076. Effectively flat;
  this question type is too hard for any chunk-size choice to matter
  much under the current generator.

The router's **real comparator is the majority-class strategy** (always
pick 128), which already achieves the best fixed-size F1 of 0.213 on
test. Any router worth shipping has to beat that by a margin large
enough to justify the routing apparatus, *and* it has to do that on a
class-imbalanced training distribution. The +8.19-point ceiling is
honest about how much absolute lift is achievable; the 82.1 %
concentration is honest about how hard it is to capture. The report
should lead with both numbers.

### How to talk about this in the report

A defensible single-paragraph framing for the methods section:

> The router's action space is restricted to {128, 256, 512}. Chunk size
> 1024 is dominated on this corpus along the dimensions that matter for
> routing: it has the worst single-size baseline F1 (0.157, vs. 0.213 for
> 128) and is the rarest per-question oracle winner (≈ 6 % on test, 5 %
> on train, 9 % on validation). Including it would require a calibrated
> 4-way classifier to learn a low-prior class whose best-case contribution
> is dominated by the next-best size on average. Restricting to three
> sizes reduces the test-set oracle gap from +10.61 to +8.19 F1 points
> overall, with the loss concentrated in multihop and synthesis questions
> (−2.80 and −3.27 points respectively); factoid is essentially unchanged
> (−1.19 points). Under the restricted action space, size 128 wins the
> per-question oracle on 82.1 % of test questions, so the router's hard
> baseline is the majority-class strategy "always pick 128" (test
> F1 = 0.213); the +8.19-point oracle ceiling sits above that baseline
> and is concentrated in factoid (+14.63 pts), with multihop and
> synthesis contributing little (+3.02 and +2.05 pts). We retain the
> 1024 chunks and FAISS index in the artifact tree so the ablation that
> motivates this choice (`artifacts/baselines/*_full.json`) remains
> exactly reproducible.

A defensible single-paragraph framing for the discussion / limitations
section:

> Two structural facts limit how much of the +8.19-point oracle ceiling
> a learned router can plausibly capture on this corpus. First, the
> oracle distribution is dominated by size 128 (82.1 % of test
> questions), so the router has to beat a strong majority-class default
> rather than learn a balanced 3-way decision; the routable headroom is
> concentrated in factoid questions, where the per-type oracle gap is
> +14.63 F1 points on top of a non-degenerate fixed-size baseline (best
> 0.486 vs. oracle 0.633). Second, restricting the action space to three
> sizes leaves measurable headroom on the table for multihop and
> synthesis questions (≈ half their full-grid per-type oracle gap),
> reflecting that very long contexts sometimes are the right answer for
> these question types on this corpus; a larger labeled set or a
> per-type action-space decision could recover this. We chose the
> simpler design because the smaller, better-studied 3-class router was
> a closer fit to the project's originality framing (calibrated cheap
> classifier vs. fusion baseline) than a brittle 4-class classifier with
> a 6 %-prior class would have been.

### How to reproduce both numbers

From `rag-chunk-routing/`:

```
# Canonical (3-size action space) — overwrites oracle_gap.json etc.
python -m experiments.run_baselines --config configs/base.yaml

# Ablation (full 4-size grid) — writes *_full.json alongside.
python -m experiments.run_baselines \
    --config configs/base.yaml \
    --restrict-sizes 128,256,512,1024 \
    --out-suffix _full
```

`run_baselines` does no retrieval and no generation; it is pure
aggregation over `eval_grid.jsonl`, so both runs together take a few
seconds. Re-running them after any change to `eval_grid.jsonl` is the
correct way to refresh the report's numbers; **no GPU work is required**
to reproduce the action-space comparison.

## Phase 3: Freeze and ablations

Goal of this phase: lock the numbers that will appear in the report.

Work to complete:

- Router ablations: isolating the contribution of each feature family,
  confusion matrix by question type, calibration plots. This is where the
  originality points in the rubric are earned.
- Faithfulness scoring pass across every prediction from every system.
- Final clean evaluation run with all bugs squashed.
- Frozen results file and the final version of the frontier figure.

Parallelizable: ablations and the faithfulness pass are independent. The
final evaluation run happens after both have landed.

Phase 3 ends with the experiments locked. No further code changes until the
report is submitted. If something is broken at freeze time, we discuss as a
team whether to fix or document the limitation. The report stage starts
immediately after freeze but is a separate working document.

## Risks we are actively watching

A short list of things that can kill the project early.

- QA generation and validation drifting past the Phase 1 gate. Mitigation:
  We start with a small batch and the manual review is done early.
- Generator too slow on available hardware. Mitigation: Setup GPU access.
- Oracle gap turning out to be small on this corpus. Mitigation: we plot the
  oracle distribution as soon as labels exist and pivot the routing target
  if the gap is under roughly ten percent.
- Router accuracy coming in low. This is not a risk, a calibrated, carefully
  analyzed negative result is publishable-quality work for this course, and
  the professor has been explicit that a clear question answered well beats
  a SOTA attempt. We frame the result honestly either way.
- Interface drift mid-project. Mitigation: the kickoff freeze, enforced by
  the five-person vote rule.

## Running the pipeline

The canonical order is encoded in the `Makefile`. Each target maps to one
script in `experiments/`. A full rebuild from the raw corpus is `make all`.
Individual stages can be rerun in isolation; see the `Makefile` for the list.

If you are on Windows without `make` installed, every target is a one-line
wrapper around a `python experiments/<script>.py --config configs/base.yaml`
call — read the `Makefile` and run the underlying command directly.

### QA generation, filtering, and validation

The QA lane runs in three stages: synthetic generation against OpenAI, an
automated filter (primary-only F1 check + LLM judge), then human
accept/reject/edit review. All three are resumable.

**One-time setup.** Copy `rag-chunk-routing/.env.example` to
`rag-chunk-routing/.env` and fill in your `OPENAI_API_KEY`. The `.env` file
is gitignored. Models, temperatures, and limits are pinned in
`configs/base.yaml` under `qa.generation`. Different stages use different
models (set under `qa.generation.models`): a cheap model for factoid
questions, a stronger model for multihop/synthesis questions, the answer
pass, and the judge. Per-type prompts live in `prompts/qa_generation_*.txt`
and the judge prompt in `prompts/judge.txt`.

The type mix in `qa.type_distribution` is roughly uniform across factoid /
multihop / synthesis. Stratification uses the largest-remainder method so
counts sum exactly to the requested batch size and the mix is deterministic
for a given seed.

**Generate.** From `rag-chunk-routing/`:

```
make qa-generate                    # tops up qa_raw.jsonl toward qa.target_count
python experiments/generate_qa.py   # equivalent without make
python experiments/generate_qa.py --limit 5   # small smoke run
```

`generate_qa` is incremental: it computes per-type deficits against
`qa.target_count` (counting both the existing `qa_validated.jsonl` and the
already-pending `qa_raw.jsonl`) and only generates the shortfall, appending
new rows to `qa_raw.jsonl`. Chunks already used in `qa_validated.jsonl` are
excluded from sampling so we don't mine the same chunk twice. Pass
`--force` to overwrite `qa_raw.jsonl` instead of appending, or `make
clean-qa` (`rm -rf artifacts/qa`) first.

**Why the QA set is 398 and not 400.** Earlier rounds of `generate_qa` /
`filter_qa` (before 2026-05-01) advanced the qa_id counter by the *produced*
pair count rather than the *attempted* count. When iterations were skipped
(parse error, judge rejection, missing field), the next call's offset
collided with a slot the previous call had already used, producing two
distinct questions sharing the same qa_id. This affected `qa_0256` and
`qa_0257`. After fixing the allocator (`rag_cr/qa_gen.py`: ids are now drawn
from a collision-skipping iterator seeded with every qa_id already on
disk), we de-duplicated `qa_validated.jsonl` by keeping the version of each
collided pair whose content matches the existing oracle eval grid, so the
~15 min A100 oracle pass did not need to be re-run. Net effect: 2 multihop
pairs were dropped, and split totals are 237 / 77 / 84 = 398. The test
split was unaffected by the dedup, so the +10.6 F1 oracle gap headline
stands. The audit trail is in `artifacts/splits/manifest.json` (see the
`note` field) and the regenerated SHA matches the deduped file.

**Filter.** From `rag-chunk-routing/`:

```
python experiments/filter_qa.py                 # filter + per-type top-up loop
python experiments/filter_qa.py --no-topup      # single filter pass, no generation
```

The filter does two things to every pending pair in `qa_raw.jsonl`:

1. **Primary-only F1 check (multihop/synthesis only).** Re-asks the answer
   model with only the primary chunk (no neighbors). If the resulting
   answer's token-F1 against the gold answer is at or above
   `qa.generation.primary_only_f1_threshold`, the pair is dropped — the
   question evidently didn't actually require the neighbors, so it isn't
   really multihop/synthesis.
2. **LLM judge.** A separate judge model sees the question, gold answer,
   primary chunk, neighbors, and claimed type, and decides keep or drop.

Pairs the judge keeps move to `qa_validated.jsonl` (still subject to human
review). Filter-rejected pairs go to `qa_rejected.jsonl` and are removed
from `qa_raw.jsonl`. With top-up enabled (the default), the script then
loops up to `qa.generation.max_topup_rounds` times: each round it generates
just enough new pairs to cover any per-type deficit, filters them, and
stops once every type hits quota.

**Validate.** From `rag-chunk-routing/`:

```
make qa-validate                    # interactive human review
python experiments/validate_qa.py   # equivalent without make
```

The reviewer sees the primary chunk, plus neighbors (for multihop /
synthesis pairs, using `qa.neighbor_window`), the question, and the gold
answer. Controls per pair: `a` accept, `r` reject, `e` edit then accept,
`q` quit and save progress. As decisions land, pairs are removed from
`qa_raw.jsonl`; rejects are also removed from `qa_validated.jsonl` and
their `qa_id` appended to `qa_rejected.jsonl`. Re-running picks up where
you stopped.

**Outputs.** All three files live in `artifacts/qa/` and are committed to
the repo (the only carve-outs to `artifacts/` being gitignored, because
hand-validated QA is not rebuildable):

- `qa_raw.jsonl` — pairs awaiting human review. Each row has been generated
  and judge-passed but not yet seen by a human. Schema: `qa_id`,
  `question`, `answer`, `source_chunk_id`, `type`, `validated=False`.
- `qa_validated.jsonl` — pairs the judge passed and (eventually) the human
  also accepted, possibly with edits. Same schema with `validated=True`.
  **This is the file downstream consumers read.** Note: rows can be
  present here before human review — the human pass refines this set
  rather than building it from scratch.
- `qa_rejected.jsonl` — sidecar carrying `qa_id`, plus `type`, `filter`,
  and `reason` for filter-stage rejects. Used to skip already-rejected
  items on re-runs. Not consumed downstream.

**For the downstream integrator.** Read `artifacts/qa/qa_validated.jsonl`
via `rag_cr.io.read_jsonl` (or `get_qa_validated_path(artifacts_dir)`).
Each row matches `rag_cr.types.QAPair`. Filter on `validated=True` if you
want to be defensive, though every row in this file is validated by
construction.

## What to do if you are stuck

Ask on Whatsapp rather than branching away on your own. The cost of a fifteen-
minute clarifying conversation is much lower than the cost of rewriting
someone else's interface later in the project.

---

## Report and documentation files

| File | Purpose |
| --- | --- |
| `neurips_2026.tex` | NeurIPS 2026 paper template. Currently contains: fixed-size baselines + oracle table, full **Chunk-Size Router** subsection (feature extractors, two-pass selection, test results, type-aware baseline paragraph, per-type figure, analysis), and placeholder sections for Introduction / Method / Conclusion. |
| `neurips_2026.sty` | Official NeurIPS 2026 style file. Required by `neurips_2026.tex`; do not modify. |
| `REPORT_GUIDE.md` | Guide for report writers. Explains the repository layout, the three-tier data philosophy, which artifact files map to which figures, and how to reproduce every number in the paper. **Start here if you are writing the report.** |
| `LitReview.tex` | Literature review document. |
| `ML Primary Lit/` | PDF copies of the primary reference papers (RAG original, dense-passage retrieval, etc.). Not consumed by any script. |

### Committed figures and LaTeX tables (`results/figures/`)

| File | Generated by | Contents |
| --- | --- | --- |
| `table_main_results.tex` | `make_figures.py` | Fixed-size baselines + oracle (F1, EM, faithfulness) |
| `table_extended_baselines.tex` | `make_figures.py` | All 4 sizes including retired 1024, with token cost |
| `table_per_type_gap.tex` | `make_figures.py` | Oracle gap breakdown by question type |
| `table_qa_composition.tex` | `make_figures.py` | QA dataset composition by split and type |
| `table_action_space_ablation.tex` | `make_figures.py` | Oracle gap with vs. without 1024 in action space |
| `table_reproducibility.tex` | `make_figures.py` | Reproducibility checklist (seeds, versions, hashes) |
| `table_router_results.tex` | manual | Router vs. type-aware vs. baseline vs. oracle |
| `fig_oracle_f1_distribution.{pdf,png}` | `make_figures.py` | Distribution of per-question oracle F1 scores |
| `fig_size_distribution.{pdf,png}` | `make_figures.py` | Oracle-best chunk size distribution across splits |
| `fig_router_comparison.{pdf,png}` | `make_router_figures.py` | Bar chart: Oracle / Type-aware / Baseline / Router |
| `fig_router_per_type.{pdf,png}` | `make_router_figures.py` | Per-type F1 breakdown across three systems |
| `frontier.{pdf,png}` | `make_frontier.py` | Accuracy–cost frontier across all systems |

---

## Library modules: `rag_cr/`

All reusable logic lives here behind frozen interfaces. No CLI entry points — those
are in `experiments/`. The internal `rag_cr/README.md` has a one-line summary per
module; this section gives the full picture.

### Configuration and infrastructure

**`config.py`** — frozen dataclasses loaded from YAML via `load_config(path)`.
The root object is `Config`; sub-configs are:

| Dataclass | Fields | Purpose |
| --- | --- | --- |
| `PathsConfig` | `corpus`, `artifacts_dir`, `results_dir` | Canonical path roots. All scripts derive sub-paths from these. |
| `ChunkingConfig` | `sizes: list[int]`, `overlaps: dict[int,int]` | Active action space and per-size overlap. 1024 is in `overlaps` but not `sizes`. |
| `EmbeddingConfig` | `model_name`, `device`, `batch_size`, `normalize` | BGE-small embedding settings. |
| `RetrievalConfig` | `top_k`, `fusion_constant`, `top_k_by_size` | `k_for_size(size)` returns per-size top-k override or falls back to `top_k`. |
| `SplitsConfig` | `ratios`, `stratify_by` | Train/val/test split ratios and stratification key (`type`). |
| `GenerationConfig` | `backend`, `model_name`, `max_new_tokens`, `temperature` | Backend is one of `ollama`, `vllm`, `extractive`. |
| `QAConfig` | `target_count`, `type_distribution`, `generation.models`, etc. | QA generation hyperparameters, per-type model overrides, filter thresholds. |

**`io.py`** — JSONL/JSON helpers and path resolution. Every script uses these
instead of calling `open()` directly, so encoding and parent-dir creation are
handled uniformly.

| Function | What it does |
| --- | --- |
| `read_jsonl(path)` / `write_jsonl(path, rows)` | Read/write a list of dicts as newline-delimited JSON. `read_jsonl` raises `ValueError` on malformed lines. |
| `read_json(path)` / `write_json(path, data)` | Load/save a single JSON object. |
| `read_text(path)` / `write_text(path, content)` | UTF-8 text files. |
| `ensure_dir(path)` | `mkdir -p`; returns resolved `Path`. |
| `timestamp()` | Returns `YYYYMMDD_HHMMSS` string used in run-dir names. |
| `create_run_dir(results_dir, system_name)` | Creates and returns `results/runs/<timestamp>_<system_name>/`. |
| `get_chunks_path(artifacts_dir, size)` | → `artifacts/chunks/<size>.jsonl` |
| `get_index_path(artifacts_dir, size)` | → `artifacts/indices/<size>.faiss` |
| `get_qa_validated_path(artifacts_dir)` | → `artifacts/qa/qa_validated.jsonl` |
| `get_qa_raw_path` / `get_qa_rejected_path` | Same pattern for the other QA files. |

**`types.py`** — shared `TypedDict`s. Used as the canonical schema everywhere.

| Type | Key fields | Used for |
| --- | --- | --- |
| `Chunk` | `chunk_id`, `size`, `start_char`, `end_char`, `text` | One chunked passage. |
| `RetrievedChunk` | `chunk_id`, `text`, `score`, `source_size` | One retrieval hit with its relevance score. |
| `QAPair` | `qa_id`, `question`, `answer`, `source_chunk_id`, `type`, `validated` | One question–answer pair. |
| `OracleLabel` | `qa_id`, `best_size`, `scores_by_size: dict[int, float]` | Per-question oracle label. |
| `ScoreDict` | `em`, `f1`, `faithfulness`, `cost_tokens` | Evaluation scores for one prediction. |

**`tokens.py`** — `count_tokens(text) -> int` via whitespace splitting. Used by
`metrics.py` to count retrieval cost (tokens in retrieved passages) and by
`chunking.py` to enforce chunk-size targets.

**`logging.py`** — `get_logger(name)` returns a project-wide logger with consistent
`%(levelname)s | %(name)s | %(message)s` formatting. All experiment scripts call
this instead of `print` for progress messages.

**`seed.py`** — `set_seed(n)` seeds Python's `random`, NumPy, and PyTorch
(including CUDA if available). Called once at the top of every experiment script
to ensure reproducible index shuffling and CV splits.

**`utils.py`** — shared computations used by multiple experiment scripts:

| Function | Signature | What it does |
| --- | --- | --- |
| `git_hash()` | `() -> str` | Returns current HEAD commit hash, or `"unknown"` on failure. Written to `meta.json` in every run. |
| `read_oracle_gap(artifacts_dir)` | `(Path) -> (float\|None, float\|None)` | Returns `(oracle_f1_mean, best_baseline_f1)` from `oracle_gap.json`. |
| `gap_closure(router_f1, baseline_f1, oracle_f1)` | `(float, float\|None, float\|None) -> float\|None` | `(router - baseline) / (oracle - baseline)`. Returns `None` if references missing or denominator ≈ 0. |

### Corpus → index → retrieval

**`corpus.py`** — `load_corpus(path)` reads `data/corpus.txt` and returns a list
of document strings. Accessor helpers (`get_doc_id(chunk_id)`, etc.) parse the
`<docid>_<offset>` chunk ID convention used throughout.

**`chunking.py`** — `chunk_corpus(corpus, size, overlap)`. Tokenizer-aware: splits
on whitespace tokens, enforces `size` as a hard token ceiling, and slides with
`overlap` tokens of context carry-over. Deterministic for a given seed. Output
rows match the `Chunk` TypedDict and are written to `artifacts/chunks/<size>.jsonl`
by `build_indices.py`.

**`embedding.py`** — `embed_texts(texts, model_name, device, batch_size, normalize)`
using `sentence-transformers`. BGE-small (`BAAI/bge-small-en-v1.5`) is the default.
A module-level model cache means the model is loaded once per process, not once per
call.

**`indexing.py`** — FAISS flat L2 index operations:
- `build_index(embeddings)` — build from a numpy matrix.
- `save_index(index, path)` / `load_index(path)` — persist to / load from `.faiss`.
- `search_index(index, query_vec, k)` — return top-k distances and indices.

**`retrieval.py`** — `Retriever` class. Loads all configured per-size indices once
on first use and keeps them in memory. Single public method:

```python
retriever.retrieve(query: str, chunk_size: int | "fusion") -> list[RetrievedChunk]
```

`chunk_size="fusion"` calls all indices and merges via `fusion.rrf_fuse`. Any
integer calls the corresponding single-scale index.

**`fusion.py`** — see [Fusion track](#fusion-track-what-was-built) section above.

### QA generation and evaluation

**`qa_gen.py`** — `QAGenerator` class. Calls the OpenAI API with type-stratified
prompts from `prompts/qa_generation_*.txt`. Key behaviors:
- Computes per-type deficits against `qa.target_count`.
- Collision-skipping ID allocator: reads all existing `qa_id`s from disk and
  skips any that are already taken before assigning new ones (fixes the 398-not-400 bug).
- Incremental append to `qa_raw.jsonl` so interrupted runs resume cleanly.
- `--force` flag overwrites `qa_raw.jsonl` from scratch.

**`qa_filter.py`** — `filter_pending(qa_raw_path, config)`. Two-stage filter:
1. Primary-only F1 check (multihop/synthesis only): re-asks with only the primary
   chunk; drops pairs where the answer is reachable without neighbors.
2. LLM judge via `prompts/judge.txt`: returns `keep` or `drop` with a reason.

Passing pairs are appended to `qa_validated.jsonl`; rejected pairs go to
`qa_rejected.jsonl`.

**`splits.py`** — `make_splits(qa_pairs, ratios, stratify_by, seed)`. Largest-
remainder stratified split: counts sum exactly to the input size, and the type
distribution is preserved within each split. Returns a dict
`{split_name: list[QAPair]}`.

**`generation.py`** — answer-generation backends, selected by
`config.generation.backend`:
- `ollama` — local Ollama server (default for laptop runs).
- `vllm` — vLLM HTTP server (used on cluster). Requires `pip install -e ".[cluster]"`.
- `extractive` — returns the first sentence of the top retrieved chunk verbatim, no
  model required. Used by CI and offline unit tests.

**`metrics.py`** — deterministic evaluation, no LLM involved:
- `token_f1(prediction, gold)` — bag-of-words token overlap F1 after lowercasing and
  punctuation removal.
- `score(prediction, gold, passages)` → `ScoreDict` with `em`, `f1`, `faithfulness`,
  and `cost_tokens` (total tokens across retrieved passages).
- Faithfulness: fraction of unique prediction tokens that appear in the retrieved
  passages (proxy for hallucination rate).

**`oracle.py`** — `label_from_grid(grid_rows)` → `list[OracleLabel]`. Groups
eval-grid rows by `qa_id`, then selects the winning chunk size using the tiebreak
order: highest F1, then highest EM, then smallest chunk size. This is the
authoritative oracle-label function; all per-question oracle-best sizes come from here.

**`pipeline.py`** — `run(system_name, qa_path, config, limit)`. Orchestrates the
full retrieve→generate→score loop for one system over a QA set. Calls
`build_system(system_name, config)`, iterates over QA pairs, scores each prediction,
and writes `metrics.json` + `predictions.jsonl` to a timestamped run directory.

**`systems.py`** — see [System abstractions](#system-abstractions-rag_crsystemspy)
section above.

**`router/`** — see [Router track: library modules](#library-modules-rag_crrouter)
section above.

---

## `results/runs/` directory schema

Every evaluation script writes its outputs to a timestamped directory under
`results/runs/`. The directory name encodes the timestamp and system tag:

```
results/runs/<YYYYMMDD_HHMMSS>_<system_tag>/
```

| System tag | Created by | GPU |
| --- | --- | --- |
| `router` | `run_router.py` | Yes |
| `type_router` | `run_type_router.py` | No |
| `fusion` | `run_fusion.py` | Yes |
| `fixed_<size>` | `run_baselines.py` (when run in eval mode) | Yes |

### Files in every run directory

**`meta.json`** — written by every script except `run_type_router.py`:

```json
{
  "git_hash": "abc123...",
  "config_path": "configs/cluster.yaml",
  "seed": 42,
  "system": "router",
  "timestamp": "20260502_010445"
}
```

**`metrics.json`** — aggregate metrics for the run. Core fields present in all runs:

| Field | Type | Description |
| --- | --- | --- |
| `mean_f1` | float | Mean token-overlap F1 across all scored questions |
| `mean_em` | float | Mean exact match |
| `mean_faithfulness` | float | Mean faithfulness score |
| `mean_cost_tokens` | int | Mean tokens in retrieved passages per query |
| `n_test_examples` | int | Number of questions evaluated |
| `gap_closure_fraction` | float\|null | (system F1 − best baseline) / (oracle − best baseline) |

Additional fields in router/type_router runs:

| Field | Type | Description |
| --- | --- | --- |
| `policy` | str | `"type_aware"` (type_router) or `"router"` |
| `type_best_sizes` | dict | Per-type winning chunk size, e.g. `{"factoid": 128, ...}` |

**`predictions.jsonl`** — one row per question. Core fields:

| Field | Type | Description |
| --- | --- | --- |
| `qa_id` | str | Question identifier |
| `question` | str | The question text |
| `gold` | str | Gold-standard answer |
| `predicted_answer` | str | Model-generated answer |
| `f1` | float | Token-overlap F1 for this question |
| `em` | float | Exact match for this question |
| `faithfulness` | float | Faithfulness score for this question |
| `cost_tokens` | int | Tokens in retrieved passages |

Router runs additionally include:

| Field | Type | Description |
| --- | --- | --- |
| `predicted_chunk_size` | int | Chunk size selected for this question |
| `question_type` | str | `factoid`, `multihop`, or `synthesis` |

The most recent run for each tag is picked up automatically by `make_frontier.py`
and `make_router_figures.py` using the `_latest(results_dir, tag, filename)` helper
(with `_RUN_RE` regex guard to prevent `*_router` matching `*_type_router`).

---

## System abstractions: `rag_cr/systems.py`

All evaluation scripts go through a common `System` protocol so the pipeline
loop is identical regardless of which system is under evaluation.

| Class | Description |
| --- | --- |
| `FixedSizeSystem(size, config)` | Retrieves from the single FAISS index for the given chunk size. |
| `FusionSystem(config)` | Retrieves from all three indices and merges via RRF. |
| `OracleSystem(config)` | Looks up the oracle-best chunk size per question from `labels.jsonl`, then delegates to `FixedSizeSystem`. |
| `RouterSystem(pkl_path, config)` | Loads the trained router pickle, predicts a chunk size from the question text and type, then delegates to `FixedSizeSystem`. |

`get_system(name, config)` returns the right instance by name string (`"fusion"`, `"oracle"`, `"router"`, or a raw integer size). This is what `pipeline.run()` and `make_frontier.py` call internally.

---

## Frontier figure: `make_frontier.py`

The frontier is the headline output of the project. It plots every system on an
**accuracy–cost plane** (mean F1 on the y-axis vs. mean tokens per query on the
x-axis) and computes the gap-recovery-per-cost efficiency metric.

### Headline metric

```
gap_recovery_per_cost = (sys_F1 − best_baseline_F1)
                      / (oracle_F1 − best_baseline_F1)
                      / mean_cost_tokens_per_query
```

This is the number reported in the report's introduction: how much oracle gap does
a system recover per token of retrieval cost?

### Data sources (priority order)

For each system, `make_frontier.py` looks for the most recent run in
`results/runs/` and falls back to pre-computed artifacts if no run exists:

- **Fixed baselines** — `results/runs/<ts>_fixed_<size>/metrics.json`, falling
  back to `artifacts/baselines/test_summary.json` + `eval_grid.jsonl` for token costs.
- **Oracle** — `artifacts/baselines/oracle_gap.json` (F1) + `eval_grid.jsonl`
  (cost per question at its oracle-best size).
- **Fusion, Router** — `results/runs/<ts>_fusion/` and `<ts>_router/`
  respectively. Missing systems are skipped with a warning.

### Outputs

- `results/figures/frontier.{pdf,png}` — the accuracy–cost scatter plot.
- `results/runs/SUMMARY.md` — markdown table of all systems with F1, EM, token cost,
  and gap-recovery-per-cost, ready to paste into the report.

Run as part of `make frontier` (depends on `baselines`, `fusion`, `router`).

---

## Fusion track: what was built

The RRF fusion baseline retrieves from all three chunk-size indices simultaneously
and merges results via Reciprocal Rank Fusion before generating an answer.

### Library module: `rag_cr/fusion.py`

Implements `rrf_fuse(ranked_per_size, fusion_constant, top_k)`. Scores each chunk
as `Σ 1 / (fusion_constant + rank_i)` summed across all per-size ranked lists,
then returns the top-k unique chunks. Parameters come from `config.retrieval`
(`fusion_constant = 60`, `top_k = 5`).

### Experiment script: `experiments/run_fusion.py`

Runs the full retrieve→generate→score pipeline with the fusion system on the
test split. Prints a sanity check comparing fusion F1 against the best fixed-size
baseline and the oracle. Writes to `results/runs/<ts>_fusion/metrics.json`.
Runs as part of `make fusion` (depends on `oracle`).

### SLURM scripts (`slurm/`)

All cluster jobs use `account=3404007`, `partition=stud`, `qos=stud`. Logs go to
`logs/%x_%j.{out,err}` — create the `logs/` directory before submitting.

| Script | Job | GPU | Time | Notes |
| --- | --- | --- | --- | --- |
| `slurm/build_indices.slurm` | Build FAISS indices | No | 30 min | Uses `base.yaml` (CPU embeddings). Run once before any eval job. |
| `slurm/oracle_test.slurm` | Oracle eval — test split | Yes (1×A100) | 45 min | Runs `compute_oracle --splits test` then `run_baselines`. Produces the headline test numbers. |
| `slurm/oracle_full.slurm` | Oracle eval — train+val | Yes (1×A100) | 45 min | Runs `compute_oracle --splits train,val` then `run_baselines`. Appends rows to the existing `eval_grid.jsonl`; test rows from `oracle_test.slurm` are preserved. |
| `slurm/fusion.slurm` | RRF fusion eval | Yes (1×A100) | 45 min | Uses `cluster.yaml` (vLLM + Qwen2.5-7B). |

Recommended submission order:

```bash
sbatch slurm/build_indices.slurm   # once, CPU-only
sbatch slurm/oracle_test.slurm     # test split oracle + baselines
sbatch slurm/oracle_full.slurm     # train/val oracle (appends to same eval_grid)
sbatch slurm/fusion.slurm          # after indices and oracle exist
```

---

## Data, pre-built artifacts, and prompts

### Corpus: `data/corpus.txt`

The raw text corpus. Plain UTF-8, one document per line. Never written to by any
script — all derived data goes to `artifacts/`. If you change this file, delete
`artifacts/chunks/`, `artifacts/indices/`, and `artifacts/oracle/` and rebuild
with `make all`. The corpus is small enough to commit directly; do not gitignore it.

### Pre-built artifacts

All files under `artifacts/` are committed so the project is immediately runnable
without a GPU. They can be wiped with `make clean-artifacts` and rebuilt from
`make all`.

**Chunks** (`artifacts/chunks/`)

| File | Contents |
| --- | --- |
| `128.jsonl` | All chunks of width 128 tokens, with `chunk_id`, `text`, `doc_id`, `token_count`. |
| `256.jsonl` | Same, width 256. |
| `512.jsonl` | Same, width 512. |
| `1024.jsonl` | Same, width 1024 (retired from action space; kept for the ablation). |

**FAISS indices** (`artifacts/indices/`)

| File | Contents |
| --- | --- |
| `128.faiss` | Flat L2 FAISS index of BGE-small embeddings for 128-token chunks. |
| `256.faiss` | Same for 256-token chunks. |
| `512.faiss` | Same for 512-token chunks. |
| `1024.faiss` | Same for 1024-token chunks (retained for ablation reproducibility). |

**Oracle** (`artifacts/oracle/`)

| File | Contents |
| --- | --- |
| `eval_grid.jsonl` | One row per `(qa_id, chunk_size)` cell: `qa_id`, `split`, `type`, `chunk_size`, `f1`, `em`, `faithfulness`. 398 questions × 4 sizes = 1 592 rows. The authoritative source for all downstream aggregation. |
| `labels.jsonl` | Oracle label per question under the 3-size action space: best `chunk_size` by F1 (tiebreak EM then smaller size), plus `oracle_f1`, `oracle_em`. |
| `labels_full.jsonl` | Same under the full 4-size action space (ablation). |

**Baselines** (`artifacts/baselines/`)

| File | Contents |
| --- | --- |
| `oracle_gap.json` | Canonical oracle-gap summary: `best_baseline_f1`, `oracle_f1_mean`, per-type breakdowns. Used by `make_router_figures.py` and `make_frontier.py`. |
| `oracle_gap_full.json` | Same for the full 4-size action space. |
| `test_summary.json` | Per-size mean F1/EM/faithfulness on the test split, 3-size action space. |
| `test_summary_full.json` | Same, 4-size action space. |
| `size_distribution.json` | Oracle-best-size distribution per split, 3-size action space. |
| `size_distribution_full.json` | Same, 4-size. |

**QA** (`artifacts/qa/`)

See "QA generation, filtering, and validation" section above for the full schema.
`qa_validated.jsonl` (398 rows) is the only file read by downstream scripts.
`qa_rejected.jsonl` is a sidecar audit log; `qa_raw.jsonl` is empty once all
pending pairs have been reviewed.

**Splits** (`artifacts/splits/`)

| File | Contents |
| --- | --- |
| `train.jsonl` / `val.jsonl` / `test.jsonl` | Stratified splits of `qa_validated.jsonl` (237 / 77 / 84 rows). Schema: full `QAPair` plus `split` field. |
| `manifest.json` | Split metadata: counts per split/type, SHA256 of the source `qa_validated.jsonl`, and a `note` field explaining the 398-not-400 dedup. |

**Router** (`artifacts/router/`)

Produced by `make router` / `experiments/train_router.py`. Committed so that
`run_router.py` can be run without repeating the GPU-intensive training.

| File | Contents |
| --- | --- |
| `best.pkl` | Pickle of the winning `(FeatureExtractor, RouterModel)` pair, fitted on the full training set. Loaded by `run_router.py` and `RouterSystem`. |
| `best_config.json` | Human-readable record of the winning configuration: feature set, classifier name, val macro-F1, val RAG F1 from the two-pass selection procedure. |
| `cv_results.csv` | Full 3×3 CV grid results table (one row per `(feature_set, classifier)` pair): mean val macro-F1 across 5 folds, std, then val RAG F1 from the re-ranking pass. Used for Table 3 in the report. |

### Prompts: `prompts/`

Template files used by the OpenAI-backed generation and judging steps.

| File | Used by | Purpose |
| --- | --- | --- |
| `qa_generation_factoid.txt` | `generate_qa.py` | System + user prompt for generating factoid QA pairs from a chunk. |
| `qa_generation_multihop.txt` | `generate_qa.py` | Prompt for multihop questions requiring multiple chunks. |
| `qa_generation_synthesis.txt` | `generate_qa.py` | Prompt for synthesis questions requiring the whole document context. |
| `qa_answer.txt` | `filter_qa.py` | Prompt for the primary-only F1 re-answering step (multihop/synthesis check). |
| `judge.txt` | `filter_qa.py` | System + user prompt for the LLM judge that decides keep/drop on each QA pair. |
| `answer.txt` | `generation.py` (pipeline) | Prompt used at evaluation time: given retrieved chunks + question, produce the answer. |

All prompts use `{variable}` placeholders. The scripts load them via
`rag_cr.io` and substitute at call time. **Do not edit prompt files without
re-running the affected pipeline stage**, as output format changes can break
downstream JSON parsing.

---

## CI pipeline

`.github/workflows/ci.yml` runs on every push to `main` and every PR targeting `main`.

**Steps:**
1. `ruff check .` — lint the entire package
2. `pytest` on the torch-free subset: `test_metrics`, `test_oracle`, `test_config`, `test_splits`, `test_fusion`, `test_io`, `test_utils`, `test_corpus`

Torch-dependent tests (`test_chunking`, `test_retrieval`, `test_router`, `test_qa_*`) are excluded from CI because they download large HuggingFace models at first run. Run them locally with `make test`.

### Configs

| Config | Purpose |
| --- | --- |
| `configs/base.yaml` | Canonical config for local runs. Generation backend: `ollama`. Action space: `{128, 256, 512}`. |
| `configs/cluster.yaml` | Cluster override. Swaps backend to `vllm` + Qwen/Qwen2.5-7B-Instruct. All other values identical to `base.yaml`. |
| `configs/eval_dry_run.yaml` | Offline/CI config. Backend: `extractive` (no GPU, no external server). Includes all 4 chunk sizes so integration tests cover the full grid without vLLM. |

### Package configuration: `pyproject.toml`

`pyproject.toml` is the single source of truth for the package. Key sections:

- **`[project]`** — name (`rag-chunk-routing`), version, `requires-python = ">=3.11"`, and the core dependency list with version ranges compatible with the cluster `ragcr` conda environment.
- **`[project.optional-dependencies]`** — `dev` extras (pytest, ruff, black) and `cluster` extras (`vllm>=0.6,<1`). Install with `pip install -e ".[dev]"` locally or `pip install -e ".[cluster]"` on the cluster.
- **`[tool.ruff]`** — line length 100, Python 3.11 target, rules E/F/I/UP/B. `make_figures.py` and `make_frontier.py` are exempted from E501 because they contain intentionally wide LaTeX format strings.
- **`[tool.pytest.ini_options]`** — `testpaths = ["tests"]`, `addopts = "-ra"`, and the two custom markers (`slow`, `integration`).

### Dependency management

`requirements.lock` is a pinned snapshot of the full dependency tree (generated with `pip freeze`). Use it to reproduce the exact environment that produced the committed results:

```bash
pip install -r rag-chunk-routing/requirements.lock
```

---

## Completed experiments and final results

All numbers below are on the **test split (n = 84)**, action space {128, 256, 512},
evaluated with Qwen/Qwen2.5-7B-Instruct on an A100 GPU (vLLM backend).
Sources: `artifacts/baselines/`, `results/figures/`.

### Fixed-size baselines and oracle ceiling

| System | F1 | EM | Faithfulness | Mean tokens |
| --- | ---: | ---: | ---: | ---: |
| Fixed (size 128) | 0.2128 | 0.1429 | 0.3626 | 1 651 |
| Fixed (size 256) | 0.1701 | 0.0714 | 0.3635 | 1 654 |
| Fixed (size 512) | 0.1695 | 0.0833 | 0.3140 | 1 638 |
| Fixed (size 1024, retired) | 0.1573 | 0.0714 | 0.3092 | 1 962 |
| **Oracle ceiling** | **0.2947** | — | — | — |

The oracle assigns the best chunk size per question individually (by F1, tiebreaked by EM then smaller size). The oracle–baseline gap is **+8.19 F1 points**.

### Per-type breakdown (oracle gap, test split)

| Question type | n | Best baseline F1 | Oracle F1 | Gap |
| --- | ---: | ---: | ---: | ---: |
| Factoid | 28 | 0.486 (size 128) | 0.633 | **+14.63 pts** |
| Multihop | 28 | 0.125 (size 256) | 0.155 | +3.02 pts |
| Synthesis | 28 | 0.076 (size 512) | 0.097 | +2.05 pts |

Factoid questions dominate both in count and in routable headroom. Multihop and synthesis have a small absolute gap and low absolute F1, meaning the routing problem on this corpus is effectively a *majority-class default (128) with selective factoid override*, not a balanced 3-way decision.

### Router and sanity baseline

| System | Mean F1 | Clf macro-F1 | Gap closure |
| --- | ---: | ---: | ---: |
| Fixed baseline (size 128) | 0.213 | — | 0.00 |
| Type-aware heuristic | 0.229 | — | **+0.20** |
| Trained router (MiniLM + LR) | 0.171 | 0.292 | **−0.51** |
| Oracle ceiling | 0.295 | — | 1.00 |

Gap closure = (router F1 − best baseline F1) / (oracle F1 − best baseline F1).

**The trained router is a negative result.** It scores below the best fixed baseline. Three compounding causes:

1. **Small training set** (237 examples). Classification macro-F1 drops from 0.416 on validation to 0.292 on test — the model does not generalise.
2. **Narrow oracle gap** (+8.19 F1 points). Routing errors are cheaper to make than gains are to earn; even a few misrouted factoid questions erase the potential lift.
3. **Type is the only useful signal.** The parameter-free type-aware heuristic — route each question to its type's best-on-average chunk size with no training — outperforms both the trained router and the fixed baseline. This confirms that the learned features (TF-IDF bigrams, MiniLM embeddings, handcrafted query features) capture nothing beyond what question type already encodes.

---

## Router track: what was built

### Library modules (`rag_cr/router/`)

| Module | Purpose |
| --- | --- |
| `features.py` | Three feature extractors behind a common `FeatureExtractor` interface: `TfidfExtractor` (bag-of-bigrams, up to 10 000 features), `MiniLMExtractor` (frozen `all-MiniLM-L6-v2` embeddings, 384-d, module-level singleton cache), `HandcraftedExtractor` (13-d deterministic: query length, question-word one-hot, heuristic NER count, question-type one-hot). Also `ConcatExtractor` for horizontal stacking. |
| `models.py` | Three classifier wrappers behind a common `RouterModel` interface: `LogisticRouter` (sklearn LR with `class_weight="balanced"`), `SVMRouter` (linear SVM with CalibratedClassifierCV; falls back to softmax if any class has fewer than 3 samples), `LightGBMRouter` (LightGBM with `is_unbalance=True`, DataFrame-name warning suppressed). All implement `predict`, `predict_proba`, and `pickle`-roundtrip. |
| `train.py` | `load_router_data` — loads (questions, types, oracle labels) from a split file, filters to the active action space. `run_cv_grid` — stratified k-fold CV across every (feature_set, classifier) pair; returns a tidy DataFrame. `rank_all_on_val` — re-fits each grid cell on the full training set and evaluates on the val set; returns a ranked list sorted by val macro-F1. |

### Experiment scripts (`experiments/`)

| Script | What it does | GPU needed |
| --- | --- | --- |
| `train_router.py` | Runs the full 3×3 CV grid (TF-IDF / MiniLM / Handcrafted × LogReg / SVM / LightGBM) with 5-fold stratified CV. Prints a CV summary table. Then applies **two-pass val selection**: top-3 cells by val macro-F1 are re-ranked by end-to-end val RAG F1 (full pipeline with predicted sizes); winner selected by RAG F1. Saves `artifacts/router/best.pkl` and `best_config.json`. | Yes (MiniLM embed + vLLM for RAG re-ranking) |
| `run_router.py` | Loads `artifacts/router/best.pkl`, predicts chunk sizes on the test split, runs the full RAG pipeline for each prediction, computes F1/EM/gap-closure. Writes `results/runs/<ts>_router/metrics.json` and `predictions.jsonl`. | Yes (vLLM) |
| `run_type_router.py` | Parameter-free sanity baseline. Reads the pre-built eval grid, computes per-type mean F1 by size, assigns each test question its type's best size, looks up F1 directly from the grid — **no new inference**. Writes `results/runs/<ts>_type_router/metrics.json` and `predictions.jsonl`. | No |
| `make_router_figures.py` | Generates two figures: `fig_router_comparison.{pdf,png}` (bar chart: Oracle / Type-aware / Best baseline / Trained router) and `fig_router_per_type.{pdf,png}` (grouped bars per question type). Reads oracle_gap.json, router and type_router metrics/predictions. | No |

All four scripts run as part of `make router`.

### Two-pass model selection

Standard selection by val macro-F1 alone is insufficient when the training distribution is imbalanced (82 % of oracle labels are class 128): a classifier that mostly predicts 128 can score high macro-F1 on val while harming the end-to-end RAG F1. The two-pass procedure fixes this:

1. Run the full CV grid. Aggregate mean macro-F1 per (feature_set, classifier) cell.
2. Re-fit each of the top-3 cells on the full training set, evaluate on val.
3. Sort those 3 by **val RAG F1** (run the full pipeline with predicted sizes). Select the winner.

Selected model: **MiniLM + logistic regression** (val macro-F1 = 0.416, val RAG F1 = 0.278).

### Key design decisions

- **Class-balanced loss everywhere.** `class_weight="balanced"` for LogReg and SVM; `is_unbalance=True` for LightGBM. Plain accuracy on val would select a model that always predicts 128.
- **TF-IDF fitted inside each CV fold.** Fitting on the full training set before CV would leak the vocabulary distribution into val folds and inflate macro-F1. `run_cv_grid` re-fits the extractor per fold.
- **`_latest()` regex guard in figures.** The glob `*_router` would also match `*_type_router` directory names. A `_RUN_RE` regex (`^\d{8}_\d{6}_(.+)$`) ensures exact tag matching.
- **No 1024 in the action space.** See the "Action space" section above for the full rationale.

### Committed artifacts

| Path | Contents |
| --- | --- |
| `artifacts/router/best.pkl` | Winning `(FeatureExtractor, RouterModel)` pickle, fitted on full train set |
| `artifacts/router/best_config.json` | Selected configuration: feature set, classifier, val macro-F1, val RAG F1 |
| `artifacts/router/cv_results.csv` | Full 3×3 CV grid results (macro-F1 per fold, val RAG F1) |
| `results/figures/fig_router_comparison.{pdf,png}` | Bar chart comparing all four systems |
| `results/figures/fig_router_per_type.{pdf,png}` | Per-type F1 breakdown (factoid / multihop / synthesis) |
| `results/figures/table_router_results.tex` | LaTeX table with router, type-aware, baseline, and oracle rows |

---

## Test suite

Run with `make test` (alias for `pytest` from `rag-chunk-routing/`).

### Coverage summary

| Test file | What it covers | # tests |
| --- | --- | --- |
| `test_config.py` | Config loading from YAML | 1 |
| `test_chunking.py` | Tokenizer-aware fixed-size chunking | ~15 |
| `test_corpus.py` | Corpus I/O and metadata | ~8 |
| `test_fusion.py` | RRF fusion logic | ~10 |
| `test_io.py` | JSONL/JSON helpers, run-dir creation | ~15 |
| `test_metrics.py` | EM, token-F1, faithfulness scorers | ~20 |
| `test_oracle.py` | Oracle label derivation, tiebreak rules | 7 |
| `test_qa_filter.py` | Primary-F1 check + LLM-judge filter (mocked) | 9 |
| `test_qa_gen.py` | QA generation pipeline (mocked) | 14 |
| `test_retrieval.py` | Single-scale and RRF retrieval | 26 |
| `test_router.py` | Feature extractors, classifiers, CV grid, val selection | 46 |
| `test_splits.py` | Stratified train/val/test split | 7 |
| `test_type_router.py` | Type-aware baseline, `_gap_closure`, `_latest`, figure smoke tests | 28 |
| `test_utils.py` | Shared utilities (gap closure, oracle gap reader) | ~10 |
| **Total** | | **≈ 167** |

### Shared fixtures: `tests/conftest.py`

`conftest.py` is loaded automatically by pytest before any test file. It adds the
repo root to `sys.path` (so imports like `from rag_cr.metrics import ...` work
without an editable install) and provides two session-scoped fixtures available to
every test file:

- `project_root` — `Path` pointing to the `rag-chunk-routing/` directory.
- `sample_corpus_text` — a short deterministic string used by unit tests that need
  a corpus without touching the real `data/corpus.txt`.

### Markers

- `integration` — requires built artifacts on disk (e.g. `test_retrieval.py` needs `artifacts/indices/` from `make indices`). Skip with `pytest -m "not integration"`.
- `slow` — loads large models (MiniLM). Skip with `pytest -m "not slow"`.

### What is not tested

Thin wrappers (`logging`, `seed`, `tokens`, `types`) and backends that require external services (`generation`, `embedding`, `indexing`, `pipeline`) have no unit tests. Experiment scripts in `experiments/` are not unit-tested; they are exercised end-to-end via the `make` pipeline. `qa_gen` and `qa_filter` are tested with the LLM call monkeypatched.

---

## Experiment scripts master table (`experiments/`)

Every script in `experiments/` is a thin CLI wrapper around library code in
`rag_cr/`. All accept `--config <path>` and write outputs to `artifacts/` or
`results/runs/<timestamp>_<tag>/`. Run via the `Makefile` targets shown below
rather than invoking directly when dependencies matter.

### Pipeline (build order)

| Script | `make` target | Needs GPU | Output |
| --- | --- | --- | --- |
| `build_indices.py` | `make indices` | No | `artifacts/chunks/*.jsonl`, `artifacts/indices/*.faiss` |
| `generate_qa.py` | `make qa-generate` | No (OpenAI API) | `artifacts/qa/qa_raw.jsonl` (appends) |
| `filter_qa.py` | `make qa-filter` | No (OpenAI API) | `artifacts/qa/qa_validated.jsonl`, `qa_rejected.jsonl` |
| `validate_qa.py` | `make qa-validate` | No | Edits `qa_validated.jsonl` in place |
| `make_splits.py` | (called by `oracle`) | No | `artifacts/splits/{train,val,test}.jsonl`, `manifest.json` |
| `compute_oracle.py` | `make oracle` | Yes (vLLM) | `artifacts/oracle/eval_grid.jsonl`, `labels.jsonl` |

### Evaluation and analysis

| Script | `make` target | Needs GPU | Output |
| --- | --- | --- | --- |
| `run_baselines.py` | `make baselines` | No | `artifacts/baselines/oracle_gap.json`, `test_summary.json`, etc. |
| `run_fusion.py` | `make fusion` | Yes (vLLM) | `results/runs/<ts>_fusion/metrics.json` |
| `train_router.py` | `make router` | Yes (MiniLM + vLLM) | `artifacts/router/best.pkl`, `best_config.json`, `cv_results.csv` |
| `run_router.py` | `make router` | Yes (vLLM) | `results/runs/<ts>_router/metrics.json`, `predictions.jsonl` |
| `run_type_router.py` | `make router` | No | `results/runs/<ts>_type_router/metrics.json`, `predictions.jsonl` |
| `make_figures.py` | `make figures` | No | `results/figures/table_*.tex`, `fig_*.{pdf,png}` |
| `make_router_figures.py` | `make router` | No | `results/figures/fig_router_*.{pdf,png}` |
| `make_frontier.py` | `make frontier` | No | `results/figures/frontier.{pdf,png}`, `results/runs/SUMMARY.md` |
