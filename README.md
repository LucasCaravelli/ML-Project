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

**Three data tiers, strictly separated.** `data/` holds the raw corpus and is
never written to. `artifacts/` holds everything derived from the corpus plus
code (chunks, indices, QA pairs, oracle labels) and can be wiped and rebuilt
at any time. `results/` holds timestamped run outputs and is never overwritten
in place. If your change breaks chunking, you delete artifacts and rebuild; 
the QA we manually validated and the results we have already committed survive 
untouched.

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

## What to do if you are stuck

Ask on Whatsapp rather than branching away on your own. The cost of a fifteen-
minute clarifying conversation is much lower than the cost of rewriting
someone else's interface later in the project.
