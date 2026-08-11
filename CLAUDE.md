# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

RecSys course project (Prof. Ga Wu) for industry partner Boardy AI — a
networking/CRM product that recommends contact introductions.

**Objective (read `docs/objective.md` first — it is canonical):** all 200 real
pairs are intros **Boardy's production system already recommended**, so every
pair passed production's own relevance bar. The label is the **real human
outcome**: the 100 positives were accepted (the two people actually connected),
the 100 negatives were declined (did not move forward). The final objective is
to train on the frozen train split and **correctly predict accept vs. decline
on the frozen holdout split** (131 train / 69 holdout real pairs, user-disjoint,
≈70/30, in `data/synthetic/seed_split.json`).

Two consequences that should shape how any result here is read: the real
negatives are **production's false positives** — plausible-looking intros that
humans still declined, so there is no easy-negative population in the real data
— and we are therefore **not modeling topical relevance** (production already
does that) but the residual "will these two actually connect." That is why
absolute AUCs sit at ~0.58–0.64, why query↔match lexical overlap is
near-identical across both classes, and why TF-IDF keyword cosine plateaus.

**Hard latency budget: <100 ms** for a user's query→retrieval round trip. This
constrains architecture more than the accuracy target does. Out at serving time:
per-candidate LLM calls, cross-encoders scoring each (seeker, candidate) pair
online, and realistically any remote embedding API on the serving path —
including `voyage-4-large` itself, whose round trip alone likely eats the budget
(it stays the accuracy *reference*, not necessarily a deployable config). In: the
two-tower/bi-encoder shape, which lets candidates be embedded offline in batch so
the online path is one query encode + an ANN lookup, flat as the pool grows — the
main architectural argument for two-tower here, independent of accuracy. A merged
LoRA adapter adds no serving cost over frozen nano, making fine-tuning the
cheapest way to buy accuracy under this budget. **No latency benchmark exists in
this repo yet** — all numbers here are offline accuracy; don't call anything a win
on accuracy alone.

Boardy's production embeddings are Voyage `voyage-4-large` (32k context), not
BERT — see `docs/boardy-embedding-model.md`. This repo benchmarks frozen
baselines (BERT, Voyage nano/large) against a LoRA-fine-tuned two-tower model,
and includes a LangGraph pipeline for synthesizing more training pairs beyond
the 200-pair seed set. Note that synthetic negatives are *constructed*
mismatches (one violated matching axis), which is not the same population as
"plausible intro a human declined" — see `docs/objective.md`.

**Status:** first full LoRA fine-tune (`twotower/`, `run_001`) did not beat
the frozen baselines on the real 69-pair holdout (pair AUC 0.578, hard-
negative-slice AUC 0.4845 — *below chance* — vs. Voyage-4-large's matched-
holdout 0.609/0.602), despite train-dev AUC of 0.986 looking dramatically
better. **Root cause found, fixed, and confirmed** (`docs/possible-bugs.md`
#4): the LoRA adapter had overfit to structural/stylistic artifacts of the
synthetic-generation prompts, not real matching semantics — a trivial
classifier shown *only* the candidate's own profile text guessed the
label with 99.2% accuracy on the old synthetic data vs. 48.7% (chance) on
real data. Fixed the generator (seed-truncation bug in `synth_pipeline/
llm.py`, banned meta-commentary give-aways, closed a style gap between
`generate_pos.md`/`generate_neg.md`) and **proved the direction is right**:
a real-only control arm (`arm_a_real_only`, 111 real pairs, zero
synthetic) beat `run_001` (530 pairs, 410 synthetic) on every metric
except pair AUC, using 1/5th the data — the unfixed synthetic data was
actively harmful, not just unhelpful. Also fixed checkpoint selection
(`possible-bugs.md` #2, was silently shipping the final epoch instead of
the best one) and rebuilt the baseline comparison on a matched population
(`docs/baseline-results-holdout.md`). **Remaining:** a full-scale
regeneration with the fixed prompts (Arm C) to see if it beats Arm A and
closes the gap to Voyage-large — scale/timing deliberately not yet
scheduled. See `docs/twotower-run-001-findings.md` (plain-language) and
`docs/twotower-run-001-results.md` (full tables) for the complete writeup.
Only a 2% sample of the 460 synthetic pairs from `batch_500_001` got real
human review before promotion — the rest were promoted on judge-verdict
alone (see "Synthetic pair review & promotion" below). **In-progress
(separate track):** a profile-first generation redesign (no label attached
at generation time, avoiding the leakage mechanism above by construction)
now has two working generators — local Ollama and AWS Bedrock — see
"Standalone profile generation" below and
`docs/profile-generation-local-and-bedrock.md`. Pairing/labeling those
profiles into a new pos/neg dataset is designed but not yet built.
**Separately: a new generalizable open-weight embedding baseline**
(`baselines/hf_embedding/`, see "Open-weight HF embedding baselines" below)
found on the 69-pair holdout that Qwen3-Embedding-8B beat Voyage-4-large
(0.6595 vs 0.6086 pair AUC). **That claim was retracted on 2026-07-31** — see
`docs/all-200-baseline-sweep.md`. Re-scored on all 200 real pairs through the
same code path, Qwen loses to Voyage-4-large on every metric (pair AUC 0.5529
vs 0.5726, MRR 0.2045 vs 0.3102, R@1 0.0500 vs 0.1300) with a below-chance
hard-negative AUC of 0.4680. **No open-weight model beats production overall**;
BGE-en-ICL is the closest, beating it on retrieval (MRR 0.3190, R@1 0.1700) but
not on AUC. The two model-loading compatibility fixes in
`docs/hf-embedding-baseline-findings.md` remain valid.

**Read `docs/baseline-results-real200.md`, not
`docs/baseline-results-holdout.md`, when comparing models.** The 69-pair
holdout ranks weak models reliably but carries no information among strong
ones: Spearman against the all-200 ranking is +0.976 across the bottom 8
models and **−0.029 across the top 6** — and every decision in this project
has been made among that top group. The all-200 population (100 positive
queries, 178-candidate corpus) is built by `eval_real_full/`.

## Experiment isolation — the rule that overrides convenience

**When trying a new idea, do not edit the previous experiment's code files. Copy
what you need into a new isolated package and edit the copy.**

Every experiment in this repo must stay a reproducible, isolated run. Editing a
shared file retroactively changes what the *earlier* experiment did — its numbers
in `docs/` can no longer be reproduced from the current tree, and that result
silently becomes unverifiable. This repo's value is largely its accumulated,
comparable measurements; that only holds if each one can still be re-run.
Duplication is cheap, a broken audit trail is not.

- **New idea → new top-level package**, not a new flag or mode bolted onto an
  existing one. Own `config.py`, own entry point, own `artifacts/<experiment>/`
  output dir. Precedents: `twotower_rrf_triplet_bigbatch/`,
  `twotower_rrf_triplet_ablation/`, `moe_reranker/`.
- **Copy, then edit** — even when the change is purely additive and
  backward-compatible. An additive edit still alters the file the earlier
  experiment ran against. (This was learned the hard way: the MoE experiment
  first added aggregation modes directly to
  `baselines/voyage_nano_sectioned/aggregate.py`, then had to be unwound.)
- **Importing shared code read-only is fine and encouraged.** Isolation means
  "don't modify", not "don't reuse" — `baselines/metrics.py` should still be the
  single source of truth for how "good" is measured, called unchanged. Prefer
  public API over private `_underscore` helpers.
- **Pin every deliberate duplicate with a test** asserting the copy still agrees
  numerically with the original, so the two cannot drift unnoticed. Example:
  `tests/test_moe_aggregation.py::test_matches_shared_baseline_on_shared_modes`.
- **Input data: copy it, never read live, never write back.** Freeze a copy into
  the experiment's own namespace with provenance — source path, content hash, and
  a `--verify` mode proving the source hasn't shifted since import. Pattern:
  `moe_reranker/import_rrf.py`. This matters most for `artifacts/pairing_rrf/` and
  `artifacts/synth/`, which are still being iterated on.
- **Log it**: a row in `docs/experiment-graphs-index.md` plus its own
  `docs/<experiment>-experiment.md` with results tables and repro commands.
- **If shared files were already edited**, restore them byte-identical to their
  pre-experiment state and verify with
  `git diff <pre-experiment-commit> -- <paths>` (expect empty output).

Orthogonal to git workflow: working directly on `main` is fine for small changes;
editing another experiment's code is not, regardless of branch.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Secrets live in `.env` (copy from `.env.example`): `OPENROUTER_API_KEY` for
synth generation, `VOYAGE_API_KEY` for the Voyage-large baseline, optional
`LANGCHAIN_API_KEY`/`LANGSMITH_*` for tracing + Prompt Hub.

## Commands

### Baseline eval (offline metrics: pair ROC-AUC/AP/best-F1, retrieval
MRR/NDCG/P/R@K, intent + neg-hardness slices — see `docs/baseline-metrics.md`)

```bash
# TF-IDF lexical baseline (no neural model — cosine similarity on TF-IDF
# vectors; establishes a lexical floor to compare the neural baselines against)
python -m baselines.tfidf.eval --data-dir data

# Frozen BERT (bert-base-uncased, mean-pool + L2 cosine; MPS if available)
python -m baselines.bert_frozen.eval --data-dir data --model bert-base-uncased --batch-size 16 --max-length 512

# Voyage-4-nano (local open-weight cousin of Boardy's prod model, same Voyage-4 space)
python -m baselines.voyage_nano.eval --data-dir data --model voyageai/voyage-4-nano --batch-size 4 --max-length 8192 --truncate-dim 1024

# Voyage-4-large (Boardy's actual production model, via API — needs VOYAGE_API_KEY)
python -m baselines.voyage_large.eval --data-dir data --model voyage-4-large --output-dimension 1024
```

Each writes an embeddings/response cache under `artifacts/<baseline>/` (so
re-runs are cheap/free) plus `artifacts/<baseline>/metrics.json`. Aggregate
all four into one exportable results file:

```bash
python scripts/export_baseline_results.py
# writes docs/baseline-results-all.json/.md and (if *_holdout artifact dirs
# exist) docs/baseline-results-holdout.json/.md — the matched-population
# comparison, see "Two-tower LoRA fine-tune" below
```

Finding so far (matched 69-pair holdout, see
`docs/baseline-results-holdout.md`): TF-IDF's pair AUC (0.592) actually
beats both twotower training runs (`run_001` 0.578, `arm_a_real_only`
0.579) — plain keyword-overlap cosine similarity, no training, edges out
the fine-tuned LoRA adapter on binary pair classification, though TF-IDF
is the weakest of all baselines on retrieval/ranking (MRR 0.248). See
`docs/possible-bugs.md` #3.

### Query ablation (searchQuery removed from seeker text)

Each baseline has a `*_no_query` sibling that packs the seeker side as
`profile_to_text(userContactFile)` only (candidate side unchanged), via
`baselines/text_no_query.py::seeker_to_text`. Used to isolate how much
signal `searchQuery` actually contributes.

```bash
python -m baselines.bert_frozen_no_query.eval --data-dir data --model bert-base-uncased --batch-size 16 --max-length 512
python -m baselines.voyage_nano_no_query.eval --data-dir data --model voyageai/voyage-4-nano --batch-size 4 --max-length 8192 --truncate-dim 1024
python -m baselines.voyage_large_no_query.eval --data-dir data --model voyage-4-large --output-dimension 1024
```

Writes to `artifacts/<baseline>_no_query/metrics.json`;
`scripts/export_baseline_results.py` picks these up automatically and adds
"(no query)" columns to `docs/baseline-results-all.md`/`.json` alongside
each baseline. Finding so far: BERT is unaffected (already near-chance, AUC
~0.47, isn't using the query either way); both Voyage models degrade
noticeably without the query, mostly on retrieval metrics (MRR/NDCG/Recall@10
down ~20–26%) — `searchQuery` is load-bearing, not redundant with the
profile.

Note: `voyage-4-nano` needs `transformers>=4.51,<5` (5.x fails to load the
remote code); requirements.txt already pins this. Cold MPS encode for nano
takes ~40+ min — subsequent runs hit the artifact cache.

### Open-weight HF embedding baselines (`baselines/hf_embedding/` + Modal)

Generic baseline for any sentence-transformers-loadable open-weight
embedding model — testing a new one is a `--model` swap, not a new
package. `models.py` holds a small per-model registry for quirks (query
prompt name, `trust_remote_code`, Matryoshka truncation, which loading
library/`transformers` version a model needs). Full findings, the
compatibility issues hit, and what's left in
`docs/hf-embedding-baseline-findings.md`.

```bash
# local (small models only, e.g. Arctic Embed / Qwen3-Embedding-0.6B)
python -m baselines.hf_embedding.eval --model Snowflake/snowflake-arctic-embed-m --holdout-only --batch-size 8 --max-length 512

# Modal GPU (models too large for local MPS, e.g. any 7-8B model)
modal run baselines/hf_embedding/modal_eval.py --model Qwen/Qwen3-Embedding-8B --holdout-only --gpu A100-40GB --batch-size 2
modal volume get dorby-hf-embedding-eval qwen_qwen3-embedding-8b_holdout ./artifacts/hf_embedding_qwen_qwen3-embedding-8b

python scripts/export_baseline_results.py   # wire a new run into docs/baseline-results-holdout.md — register it in HOLDOUT_PATHS/HOLDOUT_LABELS first
```

**Finding so far: Qwen3-Embedding-8B (Apache 2.0, free) is the first model
of any kind in this project to beat Boardy's own production model,
Voyage-4-large, on the core accept/decline task** — pair ROC-AUC 0.6595 vs
0.6086, also ahead of the previous leader (hybrid TF-IDF+nano, 0.6397).
Voyage-4-large still leads on precise top-of-list retrieval (MRR 0.5287 vs
0.4040); BGE-en-ICL turned out to be a strong retrieval model in its own
right (MRR 0.5157, 2nd-best) despite middling classification. A10G (24GB)
OOMs on any 7-8B model — use `--gpu A100-40GB`. Two models needed real
compatibility fixes beyond the standard sentence-transformers path
(NV-Embed-v2's stale `transformers` Cache API usage; BGE-en-ICL needing
BAAI's own `FlagEmbedding` library instead) — both are now handled via
`ModelSpec.requires_legacy_transformers` / `ModelSpec.loader`, see the
findings doc for the full story before trusting NV-Embed-v2's number,
which is a documented approximation. 6 open-weight models tested so far;
not every one wins — `zeroentropy/zembed-1-embedding` (4B, purpose-built
for retrieval) scored near chance (0.5052 AUC, worst retrieval MRR of
anything tested), a genuine result with no plumbing issue found.

### LLM judge (`baselines/llm_judge/`)

A different shape of experiment from every baseline above: no embeddings, no
training, no retrieval. Feed an LLM both **complete** profiles of a real pair
and ask directly whether the intro is a good match — **and never give it the
`searchQuery`**, which all other baselines get. Scored against the real human
accept/decline outcome on the 200 real seed pairs. Full writeup in
`docs/llm-judge-experiment.md`.

```bash
python -m baselines.llm_judge.eval --data-dir data --variant naive --split all
python -m baselines.llm_judge.eval --data-dir data --variant naive --split holdout  # free: shared cache
python -m baselines.llm_judge.eval --data-dir data --variant calibrated --split all
python -m baselines.llm_judge.eval --data-dir data --variant structured_cot --split holdout
python -m baselines.llm_judge.eval --data-dir data --model anthropic/claude-sonnet-4.5
# Bedrock backend instead of OpenRouter (tf_provisioner AWS account, no OpenRouter credits needed)
python -m baselines.llm_judge.eval --data-dir data --backend bedrock --model google.gemma-3-27b-it
python -m baselines.llm_judge.eval --data-dir data --backend bedrock --model qwen.qwen3-32b-v1:0
# from a worktree, data/ and .env live in the main checkout:
#   --data-dir /Users/harsh/Artifacts/dorby-ai/data --env-file /Users/harsh/Artifacts/dorby-ai/.env
```

**Findings: `gemini-3.1-flash-lite` (OpenRouter, ~$0.10/200 pairs) beats
Voyage-4-large — Boardy's production model — on the matched 69-pair holdout
(pair AUC 0.6358 vs 0.6086) *without* the search query, and its hard-negative
slice AUC of 0.6466 is the best of any model tested in this project. Two
cheaper open-weight judges added since via the Bedrock backend — Gemma 3 27B
and Qwen3-32B (~$0.03/200 pairs each) — land lower on overall AUC (0.5823,
0.5802) but each edge flash-lite out on hard-negative AUC alone (0.6216,
0.6224 vs 0.6466 — still behind, but a much smaller gap than the headline AUC
suggests); all four (model, framing) combinations sit in the same 0.58-0.64
band, comfortably above chance, none near Qwen3-Embedding-8B's 0.6595
ceiling.** Things worth carrying forward:

- **It is the only model where hard-neg AUC exceeds easy-neg AUC** (0.6466 vs
  0.5638); every embedding baseline drops steeply on hard negatives (TF-IDF
  0.7552→0.5017). Since the easy/hard split is token-overlap-defined, that
  inversion is direct evidence the LLM is not scoring lexical similarity — and
  hard negatives are the only population that exists in production. It is also
  the cleanest counter to `possible-bugs.md` #3: TF-IDF's 0.5922 is earned
  almost entirely on the easy slice.
- **Accuracy @ 0.5 is 0.5942, best in the table** — usable decisions with no
  fitted threshold, where the embedding baselines sit at/below chance there
  (Voyage-large 0.4348) because cosine has no calibrated decision point.
- **Adding true information hurt.** The `calibrated` variant (tells the model
  production already vetted relevance + the 50/50 base rate) *dropped* AUC to
  0.5901 — it only made the model more skeptical (yes-rate 56.5%→30.4%), not
  more discriminating. Stated `confidence` is also useless (88.6 when right vs
  88.2 when wrong) — the signal is all in the yes/no, not the confidence.
- **Forcing multi-aspect CoT scoring didn't help either.** `structured_cot`
  (score six weighted aspects with evidence, aggregate in code, see
  `baselines/llm_judge/structured.py`) scored *below* `naive` on the identical
  holdout, run back-to-back: pair AUC 0.6336 vs 0.6409, decision accuracy
  0.5507 vs 0.6087, hard-neg AUC 0.6267 vs 0.6543. Averaging six independent
  0-5 scores regresses toward the boundary (yes-rate jumped to 75.4% from
  55.1%) rather than sharpening the call. `naive` stays the labeling judge for
  the next synthetic batch. Full writeup in `docs/llm-judge-experiment.md`.

**This is not deployable** — a per-candidate LLM call is out of scope under the
<100 ms budget, and there are no retrieval metrics for it (no shared vector
space), which is why it is not merged into `docs/baseline-results-holdout.md`.
Its value is as an accuracy reference, as a **labeler** for
`synth_pipeline/pairing/` (whose current TF-IDF+nano labels are ~0.868-AUC
predictable from plain query cosine — this is the semantic judge that doc asks
for), and as a distillation target.

Verdicts cache to `artifacts/llm_judge/<backend>_<model>_<variant>/verdicts.json`
keyed by pair identity + **prompt hash**, so editing a prompt correctly
invalidates affected entries and `--split holdout` after `--split all` is a
pure cache hit. `tests/test_llm_judge.py` covers that plus the load-bearing
invariant that `searchQuery` never reaches the prompt.

**Cost-optimization incident:** an early attempt to run `google/gemini-3.6-flash`
and `openai/gpt-5.5` via OpenRouter failed partway through both times with
`402: insufficient credits` — not genuine account exhaustion, but because
neither call capped `max_tokens`. OpenRouter's credit check reserves against
the *requested ceiling* (the model's absolute max, e.g. 65536) rather than
actual usage, so a ~100-token JSON completion got rejected as unaffordable.
Both backends now take an explicit `--max-tokens` (default 600, plenty for
the verdict schema) — `synth_pipeline.llm.complete_json` gained an optional
`max_tokens` param for this (backward-compatible, defaults to `None` for
existing callers).

Bedrock model access via `baselines/llm_judge/bedrock_backend.py`
(`--backend bedrock --model <bedrock-model-id>`, e.g. `google.gemma-3-27b-it`
or `qwen.qwen3-32b-v1:0`, using the same `tf_provisioner` AWS account as
profile generation): tries Bedrock's native structured-JSON-schema output
first, falls back to plain-text-prompted JSON (parsed the same lenient way
as the OpenRouter path) on `ValidationException` — the same failure mode
`docs/profile-generation-local-and-bedrock.md` found for Llama 3.3 70B and
every Nova variant, handled here without needing to hardcode an exclusion
list. Bedrock usage across every model tested (tokens, invocations, errors,
latency, per-model $/hr) is on the `dorby-bedrock-profile-gen` CloudWatch
dashboard, regenerated via `scripts/update_bedrock_dashboard.py` — its
token/invocation/latency widgets use `SEARCH()` expressions that pick up any
new model automatically; only the cost widget needs a `MODEL_PRICING` entry.

### Synthetic pair generation (LangGraph + LangSmith)

Pipeline: train-only seed sample → generate one label → heuristic filter
(retry loop) → independent-model semantic judge → staging + batch manifest.
Full design in `data/synthetic/pipeline.md`.

```bash
# one-time: user-disjoint train/holdout split -> data/synthetic/seed_split.json
python -m synth_pipeline --init-split

# plumbing test, no LLM calls
python -m synth_pipeline --dry-run --n-pos 2 --n-neg 2 --batch-id dry_demo

# real batch (defaults: deepseek/deepseek-v4-pro generate, gemini-3.1-flash-lite judge, both via OpenRouter)
python -m synth_pipeline --n-pos 10 --n-neg 10 --batch-id batch_001

# after manual review sets human_review.verdict="yes" on staged files:
python -m synth_pipeline.promote --batch-id batch_001
```

Outputs land in `artifacts/synth/<batch_id>/{staged,dropped}/` +
`manifest.json`. `promote.py` is the only thing that writes into
`data/dataset_positive.json` / `dataset_negative.json`, and only for staged
files with human sign-off — never merge synth output into those files
directly.

Cost/scale reference (from `batch_pilot_010`, deepseek-v4-pro generate +
gemini-3.1-flash-lite judge via OpenRouter): 10 attempts → 9 staged, $0.1461
total, i.e. ~$0.015/attempt, ~90% judge-pass yield (small sample — expect
this to vary, especially on harder intent slices like `fundraise`). Scaling
to thousands of pairs for a two-tower fine-tune costs low tens to ~$150 in
API spend, but the real bottleneck is the human `human_review.verdict`
pass before `promote.py` — that scales linearly with staged volume and
isn't parallelizable the way generation is.

Prompts are pulled from LangSmith Hub when `LANGSMITH_PROMPT_OWNER` /
`SYNTH_PROMPT_*` env vars are set, else fall back to local
`synth_pipeline/prompts/*.md`. The judge prompt has two schema versions — v1
(dual verdict) and v2 (`would_be_good_intro` only, with `judge_verdict`
computed in code: pos passes iff good, neg passes iff not-good and not
easy-neg). Pin `SYNTH_PROMPT_JUDGE=<owner>/synth-judge:v2` while keeping
generators on `LANGSMITH_PROMPT_TAG=v1` — the two roles version
independently. Push local prompt changes with
`python -m synth_pipeline.push_prompts --owner <handle> --tag <tag>` (add
`--role judge --tag v2` for judge-only updates; `--dry-run` for no-API check).

Negatives are labeled with one of five `failure_mode`s (`wrong_side`,
`wrong_stage`, `wrong_role`, `geo_mismatch`, `prefs_conflict`), defined in
`synth_pipeline/prompts/generate_neg.md` — each is meant to share surface
jargon/topic with the query while violating exactly one axis of Boardy's
actual matching semantics (see `data/synthetic/strategy.md` for why: easy
negatives don't move hard-neg AUC, only topically-similar-but-wrong ones
do).

### Synthetic pair review & promotion

`batch_500_001` (497 attempts, 460 staged, 37 judge-rejected, ~93% yield)
is the first batch run at scale. Reviewing 460 pairs by hand doesn't scale,
so there's a browser-based spot-check workflow instead of editing staged
JSON files directly:

```bash
# rebuild the browser; flags a stratified 2% sample of each batch's staged
# pairs (across pos/neg and all 5 failure modes) as in_sample for review
python3 scripts/build_synth_browser.py

# serve it locally with a write-back endpoint (stdlib only, no new deps) —
# approve/reject buttons in the browser POST to /api/review, which writes
# human_review straight into the staged pair's JSON file
python3 scripts/serve_synth_review.py   # open http://localhost:8765
```

Toggle "Review sample only" in the browser to see just the flagged subset.
After reviewing, promote:

```bash
# strict gate: only pairs with human_review.verdict == "yes"
python -m synth_pipeline.promote --batch-id batch_500_001

# judge-verdict-only gate: promotes every staged pair regardless of human
# review — used for batch_500_001 after the 9-pair spot check passed
# clean (documented in docs/possible-bugs.md #1 as the one confirmed
# quality issue found), to hit the ~660-pair training target without a
# full manual pass
python -m synth_pipeline.promote --batch-id batch_500_001 --allow-unreviewed
```

`--allow-unreviewed` is a deliberate scope deviation from the documented
default ("only for staged files with human sign-off") — use it
consciously, not as the default path for future batches, and re-check
`docs/possible-bugs.md` for confirmed data-quality issues before trusting
a judge-only-promoted batch at face value.

**Those 460 pairs are now quarantined (2026-07-30). Do not train on them.**
They are known-harmful, not merely unhelpful: a classifier shown *only* the
candidate's profile text predicts their label with 99.2% accuracy
(`docs/possible-bugs.md` #4 — the generator leaked the label into the text),
and `twotower` `run_001` trained on them scored 0.4845 hard-negative AUC,
*below chance*, while `arm_a_real_only` beat it on a fifth of the data.

They are **not deleted** — removing them from `data/dataset_*.json` would
retroactively change what `run_001` trained on and make its published numbers
unreproducible, exactly what the isolation rule above exists to prevent.
Quarantine is enforced at the **loader**:
`twotower/data.py::build_split_bundle(include_synth=...)` now defaults to
`False`, as do `TrainConfig` and both training CLIs (`--include-synth` is a
new explicit opt-in; `--real-only` still works and still wins). An archived
copy with SHA-256 provenance lives in
`data/archive/batch_500_001_quarantined/` (see its `README.md`), pinned
against drift by `tests/test_quarantine_batch_500_001.py`.

Note the count trap this exposed: `data/dataset_*.json` holds **660 pairs /
1,217 contact ids**, but only **200 pairs / 297 contacts are real** (129
seekers, 178 candidates). Any "contacts in the dataset" figure that isn't
filtered on the `cmsynth*` prefix is counting quarantined synthetic data —
`docs/moe-rrf003-synthetic-training-findings.md` carried exactly that error
until it was corrected.

### Standalone profile generation (local Ollama + AWS Bedrock)

A separate, newer generation path from the pairs pipeline above — generates
fictional user profiles with **no pos/neg label attached at all**, so
nothing about "why this is a good/bad match" can leak into the profile text
(the root cause diagnosed in `possible-bugs.md` #4). Full design, findings,
and current status in `docs/profile-generation-local-and-bedrock.md`; pairing
these profiles into labeled pos/neg pairs (query generation → candidate
picking → existing judge) is sketched there but not yet built.

```bash
# local Ollama (127.0.0.1 + a remote Tailscale box), gemma3:4b
python scripts/local_gemma_profile_gen.py                     # run until Ctrl+C
python scripts/local_gemma_profile_gen.py --max-profiles 20   # bounded test

# AWS Bedrock — see model caveat below before running a real batch
export AWS_PROFILE=tf_provisioner AWS_DEFAULT_REGION=us-east-1
python scripts/bedrock_profile_gen.py --max-profiles 20

# browse either script's output (existing build_synth_browser.py only
# understands labeled pairs, not standalone profiles)
python3 scripts/build_profile_browser.py --runs-dir artifacts/local_gemma_synth --out artifacts/local_gemma_synth/_browser.html
open artifacts/local_gemma_synth/_browser.html
```

Both scripts implement the same 3-step design (periodically-refreshed style
spec + archetype list feeding continuous per-profile generation with a
discarded chain-of-thought `reasoning` field) and the same crash-resilience
fix: a failed style/archetype refresh now logs and keeps the stale spec
instead of crashing its worker thread — found because a ~6h51m unattended
local run lost its fast local endpoint to an uncaught `RuntimeError` ~10
minutes in and spent the rest of the run single-threaded on the slow
remote box, producing only 51 profiles instead of an estimated ~740.

**Bedrock model note:** live testing found that despite AWS's docs claiming
broad support, **Llama 3.3 70B and every Nova variant (Micro/Lite/Pro/
Premier) fail outright** on Bedrock's native structured-output JSON-schema
enforcement (`ValidationException` on both Converse `outputConfig` and
strict tool-calling) — but most *other* open-weight families (Mistral,
NVIDIA, Zhipu, DeepSeek, Google, OpenAI) and Claude 4.5+ genuinely work.
`bedrock_profile_gen.py` defaults `--model-id` to `google.gemma-3-27b-it`
(same family as the local `gemma3:4b` runs, just bigger; ~$1.30 for 500
profiles) — **smoke-tested 2026-07-23, 3/3 clean, ready for a real batch.**
Full cost table and live-test evidence in
`docs/profile-generation-local-and-bedrock.md`.

Known unfixed issues (both scripts, documented in detail in the doc above):
name/company collapse across unrelated archetypes (e.g. "Elias Vance" /
"SynapseFlow" reused for two unrelated personas — user explicitly declined
a name-blocklist fix), and one malformed archetype-list label
(`"X", "Y", "Z", (choose one appropriate)`) that propagated into every
downstream profile using it, since step 2's output isn't content-validated
the way step 3's is.

**Cost tracking:** `manifest.jsonl` records real per-call token usage
(profile generation *and* refresh calls) —
`python scripts/estimate_bedrock_cost.py <run_dir> --model-id google.gemma-3-27b-it`
computes actual $ from it. For live tracking, a CloudWatch dashboard
(`dorby-bedrock-profile-gen`, `us-east-1`) and a $10/month AWS Budget with
email alerts (filtered to `Service: Amazon Bedrock`) are set up directly
against the `tf_provisioner` account — see "Cost tracking" in
`docs/profile-generation-local-and-bedrock.md` for details and what to
update if the target model changes.

### Pairing standalone profiles into labeled pairs (`synth_pipeline/pairing/`)

Turns a pool of unlabeled profiles (from either generator above) into labeled
pos/neg pairs. **No LLM judge** — labels come from the TF-IDF+Voyage-nano
fusion scorer, which is the strongest pair scorer measured on the matched
holdout (AUC 0.6397 / hard-neg 0.6034, beating even Voyage-4-large).

```bash
export AWS_PROFILE=tf_provisioner AWS_DEFAULT_REGION=us-east-1
python -m synth_pipeline.pairing \
  --profile-run artifacts/bedrock_synth/run_<ts> \
  --batch-id pair_test_001 \
  --data-dir /Users/harsh/Artifacts/dorby-ai/data   # data/ is gitignored — required from a worktree

# side-by-side graph: real pairs | this batch
python scripts/build_real_pairs_graph.py \
  --compare artifacts/pairing/pair_test_001 \
  --out docs/html/pairs-comparison-graph.html

# verification add-on, run after building a batch (not a gate)
python scripts/verify_pairing_scorer.py
```

Five phases: load profiles (mint `cmsynthp…` ids, drop the CoT `reasoning`
field) → generate a `searchQuery` per profile via Bedrock (the only LLM call)
→ rank candidates against each query by TF-IDF and take a log-spaced top band
→ score with the fusion (fitted on **real train pairs only**) → stage.

**Labels are provisional, not training data.** A model trained on a batch can
at best imitate the scorer that labeled it; on hard pairs that scorer is right
~60% of the time. Three things keep this quarantined: batches live in their own
`artifacts/pairing/<batch_id>/` namespace, **nothing is promoted** into
`data/dataset_*.json`, and `manifest.json` records the labeler and thresholds.
Promoting such a batch would repeat the `batch_500_001` mistake in a new form.

Labeling always keeps a **deadband** — `pos` above, `neg` below, near-boundary
pairs written unlabeled to `excluded/` rather than given a coin-flip label.
Negatives carry **no `failure_mode`**: a scorer yields a number, not a
diagnosis, so it's left null rather than guessed.

**`--label-mode quantile` is the default, and the reason matters.** The first
test run used the real-pair threshold and labeled 164/164 pairs positive: the
synthetic score distribution (0.57–9.86) does not overlap the real-pair
threshold region (≈ −2.18) at all, because synthetic profiles are far more
homogeneous than real contacts *and* `select.py` picks the top-similarity band
by construction. Quantile mode splits each batch by its own distribution (top
`--pos-frac` 0.30 / bottom `--neg-frac` 0.30). `--label-mode absolute` keeps the
old behavior for comparison and now warns when the ranges don't overlap. Note
this changes what a label means: "better/worse than others offered to this
seeker in this batch", not "good by the real-pairs standard".

Two related hazards fixed while finding this: contact ids are now derived
deterministically from `sha256(source_run:profile_id)` (random ids broke re-runs
and the query checkpoint), and the batch embedding-cache key is content-hashed
— both `TfidfEncoder.encode()` and `VoyageNanoEncoder.encode()` return a cached
array whenever `cache_name` exists *without* checking the input texts still
match, so a fixed key silently served stale embeddings after a query regen.

Why not the existing judge: `judge_node` puts the label into the judge's own
prompt (`judge.py:56-63`), which was fine when the label came from elsewhere
and the judge only had veto power, but would have it grading its own answer key
here.

**`pair_test_001` (20 profiles, 2026-07-23): 52 pos / 52 neg / 70 excluded.**
Scorer verified exact — `scripts/verify_pairing_scorer.py` reproduces the
documented holdout AUC 0.6397 to four decimals. But the headline finding is a
limitation: **TF-IDF query cosine alone predicts the assigned label at 0.868
AUC**, so most of the label is plain lexical overlap — `select.py` ranks by
TF-IDF, then a TF-IDF-heavy labeler grades that ranking. Given
`possible-bugs.md` #3 (plain TF-IDF already beats both fine-tunes), training on
these labels would largely teach lexical overlap. That's the strongest argument
for putting a semantic judge back in the labeling path before this becomes
training data. Topology also diverges sharply from real data (80% of contacts
carry both labels vs 5% real; 5.2 vs 0.67 edges/node). Full results, the
distribution-shift finding, and remaining gaps are in
`docs/profile-generation-local-and-bedrock.md`.

### RRF pairing + LLM judge (`synth_pipeline/pairing_rrf/`) — full doc: `docs/rrf-pairing-pipeline.md`

Second-generation labeling path, replacing the fusion *scorer* used by
`synth_pipeline/pairing/` with **two independent retrieval channels plus an LLM
judge**, so retrieval and labeling come from different model families and no
scorer grades its own ranking (the 0.868-AUC circularity that undermined
`pair_test_001`).

Flow: profiles → disjoint split → one query per `lookingFor` section (Bedrock) →
Qwen3-Embedding-8B on Modal, N+1 seeker vectors (whole profile + one per section)
→ `.npy` files, then Chroma → dense top-10 ‖ BM25 top-10 → weighted RRF (dense
2:1, `rrf_k=60`) → top-5 → `google/gemini-3.1-flash-lite` judge, one call per
pair, no deadband → pos / hard-negative.

```bash
export AWS_PROFILE=tf_provisioner AWS_DEFAULT_REGION=us-east-1

# the reusable template — every knob lives in a preset, not in flags
python scripts/generate_rrf_dataset.py                                   # defaults
python scripts/generate_rrf_dataset.py --preset my_run.json              # tuned
python scripts/generate_rrf_dataset.py --profile-run <dir> --skip-generate
python scripts/generate_rrf_dataset.py --dry-run                         # plan only

# prompts are hub-only — push before any paid run
python -m synth_pipeline.pairing_rrf.push_prompts --tag v1

# browse a batch — three tabs in one self-contained file, zero network requests:
#   Pairs      leakage/circularity probes + every pair's retrieval provenance
#              and judge reasoning
#   Topology   force-directed graph, this batch beside the 200 real pairs
#   Embeddings the Qwen3 vectors in 3D (PCA), seeker anchors with their
#              lookingFor asks tethered, pos/neg pair edges
# (the other build_*_browser scripts don't understand this layout — no
# failure_mode, no human-review gate)
python scripts/build_rrf_browser.py --batch-id rrf_002
open artifacts/pairing_rrf/rrf_002/_browser.html

# degrades instead of failing: --no-real drops the comparison pane when data/
# is absent (it's gitignored), and without numpy/sklearn the embeddings tab is
# hidden while the first two still build
```

Presets live in `synth_pipeline/pairing_rrf/presets/`; the preset that produced a
batch is copied into its output so results are reproducible. Stages
(`generate_profiles`/`pairing`/`judge`/`export`) toggle independently, and
`queries.json` + the judge cache make re-runs cheap.

**First run `rrf_002`: 275 pairs (64 pos / 211 neg), $0.62 judge cost, ≈$1.40
all in.** Two findings worth carrying forward: the judge's yes-rate collapsed
from the experiment's 56.5% to 23.3% on retrieved synthetic candidates (harder,
more homogeneous population — expect ~3 negatives per positive), and a
**duplicate-pair defect** was found and fixed — a seeker's several queries can
retrieve the same candidate, and judged independently 25 keys came back labeled
both `pos` and `neg`. `fuse.deduplicate_pairs()` now enforces global
`(seeker, candidate)` uniqueness by default.

Prompts are **hub-only with no local fallback** (`prompt_hub.py`), matching
`scripts/profile_gen_prompt_hub.py`: `-/pair-rrf-query:v1`, `-/pair-rrf-judge:v1`.
Note `PROFILE_GEN_PROMPT_GENERATE` must be pinned to v3+ — unpinned it falls back
to `LANGSMITH_PROMPT_TAG=v1` and every profile dies with
`KeyError: 'ref_example_1'`.

Labels are a model's opinion, not real accept/decline outcomes. Batches stay in
`artifacts/pairing_rrf/<batch_id>/`, are exported to the git-tracked
`exports/rrf_datasets/`, and **nothing is promoted** into `data/dataset_*.json`.

**Trainability probes on `rrf_002`** (table in `docs/rrf-pairing-pipeline.md`):
the generation-artifact leak that destroyed `run_001` is gone — candidate-profile-
only prediction is 0.634 AUC, not 99.2% accuracy — and lexical circularity dropped
from `pair_test_001`'s 0.868 to 0.701. The live weakness is **per-node base rate**:
seeker identity alone, with no text at all, predicts the label at 0.687, because
**12 of 40 seekers were rejected on every candidate**. So do not train a plain
pairwise classifier on this batch; train **within a seeker**, which cancels that
base rate by construction. 28 of 40 seekers carry both classes, giving **249
(anchor, +, −) triplets** — `run_001`'s pool had 5 of 91, which is the sole reason
`two-tower-fine-tune-plan.md` picked `ContrastiveLoss` over
`MultipleNegativesRankingLoss`. That constraint is now lifted. Judge accuracy on
the hard slice is 0.5942, so any model trained here must be scored on the **real**
holdout, never on held-out synthetic pairs.

### Two-tower LoRA fine-tune (`twotower/` + Modal)

LoRA fine-tune of `voyage-4-nano` on the promoted dataset. Architecture and
loss-choice rationale (pairwise `ContrastiveLoss`, not `MultipleNegatives-
RankingLoss` triplets, because current pos/neg pairs mostly don't share a
seeker — only 5 of 91 synth seekers had both) are in
`docs/two-tower-fine-tune-plan.md`. Results, root-cause diagnosis, fixes
applied, and the Arm A/B/C experiment plan are in
`docs/twotower-run-001-findings.md` (plain-language) and
`docs/twotower-run-001-results.md` (full tables).

```bash
# local (CPU/MPS smoke-test only — MPS LoRA backward is weak, prefer Modal for real runs)
python -m twotower.train --dry-run --epochs 1

# Modal GPU (L4 default), full run
modal run twotower/modal_train.py --run-id run_001 --epochs 5
# real-only control arm (excludes all promoted synth pairs from train pool)
modal run twotower/modal_train.py --run-id arm_a_real_only --epochs 5 --real-only

modal volume get dorby-twotower-checkpoints run_001 ./artifacts/twotower/run_001
# NOTE: `modal volume get` errors "Is a directory" when downloading a
# whole run directory in one call on this CLI version — pull run_meta.json/
# run_result.json/metrics_train_dev.json and the adapter/ subfiles
# individually instead (see docs/twotower-run-001-results.md).

# holdout eval — one-time final check only, per the decision-gate rule in
# docs/two-tower-fine-tune-plan.md; do not run repeatedly while iterating
python -m twotower.eval --split holdout --adapter-dir artifacts/twotower/run_001/adapter

# matched-population comparison across BERT/nano/large/twotower runs
python -m baselines.bert_frozen.eval --holdout-only --artifacts-dir artifacts/bert_frozen_holdout ...
python scripts/export_baseline_results.py   # also builds docs/baseline-results-holdout.md
```

`twotower/data.py::build_split_bundle()` is leakage-safe by construction:
holdout = frozen `eval_pair_ids` only, train pool = frozen train pairs +
promoted synth pairs that touch no eval user (or zero synth pairs with
`include_synth=False` / `--real-only`), train-dev = a further user-disjoint
carve from train. `assert_no_holdout_leak()` is called in both `train.py`
and `eval.py`; `tests/test_twotower_data.py` covers this plus deterministic
carving, the real-only mode, and split-hash tamper rejection.

**`run_001`** (5 epochs, full 530/61/69 split): train-dev pair AUC looked
great (0.986) but real 69-pair holdout pair AUC was 0.578, hard-negative
AUC 0.4845 (*below chance*) — did not beat the frozen baselines. Root cause
confirmed (`docs/possible-bugs.md` #4): the LoRA adapter overfit to
synthetic-generation-prompt artifacts. **`arm_a_real_only`** (same recipe,
111 real pairs, zero synthetic) then beat `run_001` on every metric except
pair AUC using 1/5th the data — proof the synthetic data was actively
harmful, not just unhelpful, and that the architecture/recipe are fine on
their own. Both the data-generation root cause (`synth_pipeline/llm.py`
seed-truncation bug, `generate_neg.md` meta-commentary/tone gaps) and
checkpoint selection (`docs/possible-bugs.md` #2, was silently shipping
the final epoch) are fixed and confirmed. **Remaining: Arm C** — full-scale
regeneration with the fixed prompts, trained and compared against Arm A's
bar — scale/timing deliberately not yet scheduled, do not generate a large
batch or launch further runs until that's decided.

### Data prep utilities

```bash
python scripts/build_unique_users.py --data-dir data          # dedupe pairs -> data/unique_users.json
python scripts/build_unique_users_browser.py                  # self-contained HTML browser for that catalog

# per-field BERT token counts + filterable HTML browser (needs project venv + cached tokenizer)
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/build_unique_users_token_counts.py
python scripts/build_unique_users_token_counts.py --html-only  # rebuild HTML only, no tokenizer
```

## Architecture

### `baselines/` — offline embedding baselines

Four sibling packages (`tfidf`, `bert_frozen`, `voyage_nano`,
`voyage_large`), each with `encode.py` (model-specific embedding logic) +
`eval.py` (CLI entry: load pairs, embed, score, write `metrics.json`). All
four share `baselines/metrics.py` (pair + retrieval + slice metrics — the
single source of truth for how "good" is measured) and
`baselines/bert_frozen/text.py` (field-tagged text serialization of a
contact profile into a string, shared even by the Voyage and TF-IDF
baselines). If you touch the text serialization or metric definitions, all
four baselines are affected — keep them in sync rather than forking.
`tfidf` differs structurally from the other three: it has no pretrained
model, so `TfidfEncoder.fit()` must be called on the full corpus before any
`encode()` calls (vocabulary/IDF are corpus-dependent, unlike a frozen
neural encoder) — see `run_eval()` in `baselines/tfidf/eval.py` for the
fit-then-encode ordering.

Each of the three neural baselines has a `*_no_query` sibling package
(`bert_frozen_no_query`, `voyage_nano_no_query`, `voyage_large_no_query`)
for the query-ablation eval — same `encode.py`, same metrics, only the
seeker-text packing changes via `baselines/text_no_query.py`. See "Query
ablation" under Commands. `tfidf` doesn't have a `_no_query` sibling yet.

`baselines/holdout.py::filter_to_holdout()` is shared by all `--holdout-only`
flags (added to every baseline `eval.py`) — filters to the frozen 69-pair
`eval_pair_ids` from `data/synthetic/seed_split.json`, so baseline and
twotower holdout numbers are computed on an identical population. See
"Two-tower LoRA fine-tune" below and `docs/baseline-results-holdout.md`.

`baselines/hf_embedding/` is a fifth, differently-shaped package: instead of
one model per package, it's one generic `encode.py`/`eval.py` pair
parameterized by HF model id, for testing free/open-source embedders
beyond the Voyage family (see "Open-weight HF embedding baselines" under
Commands). `models.py`'s `MODEL_REGISTRY` holds per-model quirks; two
models needed a genuinely different loading path beyond the default
sentence-transformers one (a pinned-older-`transformers` image for
NV-Embed-v2, BAAI's own `FlagEmbedding` library for BGE-en-ICL, both
routed via `ModelSpec` fields) — see
`docs/hf-embedding-baseline-findings.md` before adding another odd model.

### `synth_pipeline/` — LangGraph synthetic pair generator

A linear graph (`graph.py`) over a single `TypedDict` state (`state.py`,
`PairState`): `seed_sampler → generate → heuristic_filter ⇄ generate (retry,
max `cfg.max_retries`) → semantic_judge → staging → END`. Node
implementations live in `synth_pipeline/nodes/` (`seed.py`, `generate.py`,
`filter.py`, `judge.py`, `writer.py`), each a plain function of
`(state, cfg) -> partial state update` — read `graph.py`'s routing functions
(`_route_after_filter`, `_route_after_judge`) first to understand control
flow before touching a node.

`config.py`'s `PipelineConfig` is the dependency-injection point for
everything (model names, paths, retry/holdout settings, prompt version) and
is threaded via `functools.partial` into every graph node — new
configuration should be added there, not read from `os.environ` inside a
node.

Train/holdout discipline is load-bearing: `split.py` computes and persists a
user-disjoint split (`data/synthetic/seed_split.json`, content-hashed as
`split_hash`), and `assert_train_only` is called both at seed-sampling time
and again inside `seed_sampler_node` to catch split drift mid-batch — never
sample seeds/few-shots from the holdout set. `ids.py` mints synthetic contact
IDs up front (before generation) matching the real ID format (`cm` + 25
chars) so the generate step is constrained to use pre-assigned IDs rather
than inventing its own.

`promote.py` is the only path from `artifacts/synth/<batch>/staged/*.json`
into the canonical `data/dataset_positive.json` / `dataset_negative.json`,
gated on `human_review.verdict == "yes"` and schema validation
(`schema.py::validate_pair_schema`) and deduped against existing
`(userContactId, matchContactId)` keys — `--allow-unreviewed` bypasses the
human-review gate (judge verdict only); see "Synthetic pair review &
promotion" under Commands.

`scripts/build_synth_browser.py` + `scripts/serve_synth_review.py`: a
review UI for staged/dropped synth pairs. The build script embeds every
batch's manifest + pair JSON into one self-contained HTML file and flags a
stratified sample (`--sample-pct`, default 2%, across pos/neg and all 5
failure modes) as `in_sample`; the serve script is a stdlib-only local HTTP
server that write-backs `human_review` verdicts from browser button clicks
straight into the staged pair's JSON file (no framework dependency added).

`synth_pipeline/pairing/` is a separate, self-contained subpackage — it shares
`config`/`ids`/`schema` with the LangGraph pipeline above but bypasses the
graph, the generators, and the judge entirely. Flow is a plain function chain,
not a graph: `profiles.py` → `query.py` → `select.py` → `label.py` →
`stage.py`, orchestrated by `run.py`. `bedrock.py` is a deliberate ~40-line
duplicate of `scripts/bedrock_profile_gen.py::call_bedrock` (a package
importing from `scripts/` would need a `sys.path` hack). Two invariants worth
knowing before touching it: `profiles.py::_extract_profile` is an allowlist so
the generator's CoT `reasoning` field can never leak downstream even if the
generation schema grows, and `select.py` enforces global uniqueness of
`(seeker, candidate)` because the pair schema has no query identity and
`promote.py` dedups on that key. `stage.ENVELOPE_KEYS` is asserted against
`nodes/writer.py`'s envelope dict by AST in `tests/test_pairing.py`, so the two
can't drift.

### `twotower/` — LoRA fine-tune of voyage-4-nano

`config.py` (`TrainConfig`, single source of truth for hyperparams/paths) →
`data.py` (leakage-safe `SplitBundle`: frozen holdout + train pool +
user-disjoint train-dev carve, `assert_no_holdout_leak`) → `train.py`
(LoRA adapter on `q/k/v/o_proj`, `SentenceTransformerTrainer` +
`ContrastiveLoss`, gradient/target-count validation before training,
`select_best_checkpoint` — currently buggy, see
`docs/possible-bugs.md` #2) → `eval.py` (reuses `baselines/metrics.py`
directly, so metric shape matches the frozen baselines exactly — same
`pair`/`retrieval`/`slices` keys). `modal_train.py` is the Modal GPU
entrypoint (L4 default, separate checkpoint + HF-cache volumes). Not a
true untied two-tower — one shared-weight model called via
`encode_query`/`encode_document` with different prompt prefixes (Voyage's
native asymmetric convention), matching how the frozen baseline already
worked; an untied/projection-head variant is documented as a fallback in
`docs/two-tower-fine-tune-plan.md` only if plain LoRA underperforms.
`tests/test_twotower_data.py` covers leakage/split-hash-tamper safety.

## Data

`data/dataset_positive.json` / `dataset_negative.json`: labeled pairs, no
`label` field — membership in the file *is* the label. Schema:
`userContactId`, `matchContactId`, `*ContactFileVersion`, `searchQuery`,
`userContactFile`/`matchContactFile` (nested profiles with `positioning`,
`background`, `lookingFor`, `notes`, `locationAvailability`,
`introPreferences`, `personalPreferences`,
`meetingAndSchedulingPreferences`). Full field-level notes in
`data/dataset_summary.md`.

Started at 100/100 real seed pairs. As of `batch_500_001` promotion
(2026-07-19), **320 positive / 340 negative (660 total)**: 100/100 real
seed + 220/240 promoted synthetic (`cmsynth*` contact IDs). Only 9 of the
460 staged synthetic pairs got real human sign-off
(`human_review.verdict == "yes"`, via the review browser below); the
remaining 451 were promoted on judge-verdict alone via `promote.py
--allow-unreviewed` — a deliberate scope call, not the documented
default gate (see "Synthetic pair review & promotion" and
`docs/possible-bugs.md`). `data/synthetic/seed_split.json` still defines
the frozen leakage-safe split: 131 real train pairs / 69 real holdout
pairs (`eval_pair_ids`) — `twotower/data.py::build_split_bundle()` is the
canonical way to load train/train-dev/holdout without leaking synthetic or
holdout-user pairs into training.
