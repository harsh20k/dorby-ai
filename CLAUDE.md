# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

RecSys course project (Prof. Ga Wu) for industry partner Boardy AI — a
networking/CRM product that recommends contact introductions. Given a user's
profile + `searchQuery`, decide if a candidate match is a good intro. Boardy's
production embeddings are Voyage `voyage-4-large` (32k context), not BERT —
see `docs/boardy-embedding-model.md`. This repo benchmarks frozen baselines
(BERT, Voyage nano/large) against a LoRA-fine-tuned two-tower model, and
includes a LangGraph pipeline for synthesizing more training pairs beyond
the 200-pair seed set.

**Status:** first full LoRA fine-tune (`twotower/`, `run_001`) has
completed and been evaluated on the real 69-pair frozen holdout — **it
does not beat the frozen baselines.** Holdout pair AUC 0.578, essentially
tied with Voyage-4-nano (0.561) and below Voyage-4-large (0.573), despite
train-dev pair AUC of 0.986 looking dramatically better. Root cause
(`docs/possible-bugs.md` #4, confirmed): the LoRA adapter appears to have
overfit to structural/stylistic artifacts of the synthetic-generation
prompts rather than learning real matching semantics — real-holdout
hard-negative-slice AUC is 0.4845, *below chance*. Per the decision gate in
`docs/two-tower-fine-tune-plan.md` ("if lift shows on train but not
holdout, stop and diagnose — don't scale data yet"), **do not generate a
larger synthetic batch or launch `run_002` until this is diagnosed** — see
`docs/twotower-run-001-results.md` for the full writeup and next-step
options. Other known open items: checkpoint selection silently picks the
final epoch instead of the best one (`possible-bugs.md` #2), and only a 2%
sample of the 460 synthetic pairs from `batch_500_001` got real human
review before promotion — the rest were promoted on judge-verdict alone
(see "Synthetic pair review & promotion" below).

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
# Frozen BERT (bert-base-uncased, mean-pool + L2 cosine; MPS if available)
python -m baselines.bert_frozen.eval --data-dir data --model bert-base-uncased --batch-size 16 --max-length 512

# Voyage-4-nano (local open-weight cousin of Boardy's prod model, same Voyage-4 space)
python -m baselines.voyage_nano.eval --data-dir data --model voyageai/voyage-4-nano --batch-size 4 --max-length 8192 --truncate-dim 1024

# Voyage-4-large (Boardy's actual production model, via API — needs VOYAGE_API_KEY)
python -m baselines.voyage_large.eval --data-dir data --model voyage-4-large --output-dimension 1024
```

Each writes an embeddings/response cache under `artifacts/<baseline>/` (so
re-runs are cheap/free) plus `artifacts/<baseline>/metrics.json`. Aggregate
all three into one exportable results file:

```bash
python scripts/export_baseline_results.py
# writes docs/baseline-results-all.json and .md
```

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

### Two-tower LoRA fine-tune (`twotower/` + Modal)

LoRA fine-tune of `voyage-4-nano` on the promoted dataset. Architecture and
loss-choice rationale (pairwise `ContrastiveLoss`, not `MultipleNegatives-
RankingLoss` triplets, because current pos/neg pairs mostly don't share a
seeker — only 5 of 91 synth seekers had both) are in
`docs/two-tower-fine-tune-plan.md`. Results and open caveats for the first
full run are in `docs/twotower-run-001-results.md`.

```bash
# local (CPU/MPS smoke-test only — MPS LoRA backward is weak, prefer Modal for real runs)
python -m twotower.train --dry-run --epochs 1

# Modal GPU (L4 default), full run
modal run twotower/modal_train.py --run-id run_001 --epochs 5
modal volume get dorby-twotower-checkpoints run_001 ./artifacts/twotower/run_001
# NOTE: `modal volume get` errors "Is a directory" when downloading a
# whole run directory in one call on this CLI version — pull run_meta.json/
# run_result.json/metrics_train_dev.json and the adapter/ subfiles
# individually instead (see docs/twotower-run-001-results.md).

# holdout eval — one-time final check only, per the decision-gate rule in
# docs/two-tower-fine-tune-plan.md; do not run repeatedly while iterating
python -m twotower.eval --split holdout --adapter-dir artifacts/twotower/run_001/adapter
```

`twotower/data.py::build_split_bundle()` is leakage-safe by construction:
holdout = frozen `eval_pair_ids` only, train pool = frozen train pairs +
promoted synth pairs that touch no eval user, train-dev = a further
user-disjoint carve from train. `assert_no_holdout_leak()` is called in
both `train.py` and `eval.py`; `tests/test_twotower_data.py` covers this
plus deterministic carving and split-hash tamper rejection.

`run_001` (5 epochs, full 530/61/69 split, `max_seq_length=4096`):
train-dev pair AUC looked great (0.986) but the **real 69-pair holdout
tells a different story: pair AUC 0.578**, essentially tied with the
frozen baselines and *below chance (0.4845)* on hard real negatives — see
`docs/twotower-run-001-results.md` for the full table and
`docs/possible-bugs.md` #4 for the root-cause writeup (likely overfitting
to synthetic-generation-prompt artifacts, not real matching semantics).
**Do not launch `run_002` or generate more synthetic data at current
settings until this is diagnosed** — that's the decision gate from
`docs/two-tower-fine-tune-plan.md` doing its job. Checkpoint selection also
has a known bug (`docs/possible-bugs.md` #2): it silently picked the final
epoch instead of the actual best-scoring one this run (epoch 3 outscored
epoch 5 on train-dev `pair_auc`, 0.989 vs 0.986) — low-impact here but
unverified at scale, fix before trusting a future run's checkpoint
selection.

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

Three sibling packages (`bert_frozen`, `voyage_nano`, `voyage_large`), each
with `encode.py` (model-specific embedding logic) + `eval.py` (CLI entry:
load pairs, embed, score, write `metrics.json`). All three share
`baselines/metrics.py` (pair + retrieval + slice metrics — the single source
of truth for how "good" is measured) and `baselines/bert_frozen/text.py`
(field-tagged text serialization of a contact profile into a string, shared
even by the Voyage baselines). If you touch the text serialization or metric
definitions, all three baselines are affected — keep them in sync rather
than forking.

Each has a `*_no_query` sibling package (`bert_frozen_no_query`,
`voyage_nano_no_query`, `voyage_large_no_query`) for the query-ablation eval
— same `encode.py`, same metrics, only the seeker-text packing changes via
`baselines/text_no_query.py`. See "Query ablation" under Commands.

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
