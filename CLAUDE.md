# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

RecSys course project (Prof. Ga Wu) for industry partner Boardy AI — a
networking/CRM product that recommends contact introductions. Given a user's
profile + `searchQuery`, decide if a candidate match is a good intro. Boardy's
production embeddings are Voyage `voyage-4-large` (32k context), not BERT —
see `docs/boardy-embedding-model.md`. This repo benchmarks frozen baselines
(BERT, Voyage nano/large) against planned two-tower / MoE / student-teacher
approaches, and includes a LangGraph pipeline for synthesizing more training
pairs beyond the 200-pair seed set.

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
`(userContactId, matchContactId)` keys.

## Data

`data/dataset_positive.json` / `dataset_negative.json`: 100/100 labeled
pairs, no `label` field — membership in the file *is* the label. Schema:
`userContactId`, `matchContactId`, `*ContactFileVersion`, `searchQuery`,
`userContactFile`/`matchContactFile` (nested profiles with `positioning`,
`background`, `lookingFor`, `notes`, `locationAvailability`,
`introPreferences`, `personalPreferences`,
`meetingAndSchedulingPreferences`). Full field-level notes in
`data/dataset_summary.md`.
