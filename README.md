# Dorby AI

RecSys course project (Prof. Ga Wu) for industry partner **Boardy AI**.

## Objective

Canonical statement: **[docs/objective.md](docs/objective.md)** — read this
first.

We were given **200 real labeled pairs**, and what they mean matters more than
the count. All 200 are introductions **Boardy's production system already
recommended** — every pair passed production's own relevance bar. The label is
the **real human outcome** of that recommendation:

| File | Records | Meaning |
| --- | --- | --- |
| `data/dataset_positive.json` | 100 | Production suggested it and the humans **accepted** — they actually connected. |
| `data/dataset_negative.json` | 100 | Production suggested it and the humans **declined** — no follow-through. |

**The final objective:** train on the frozen train split and **correctly
predict accept vs. decline on the frozen holdout split**. The split is
user-disjoint and frozen in `data/synthetic/seed_split.json` — **131 train / 69
holdout** real pairs (≈70/30; the exact ratio falls out of the no-shared-user
constraint). Holdout accuracy is the number the project is judged on.

Two things follow, and they explain most of the results below:

- **The negatives are production's false positives**, not random mismatches —
  plausible intros that humans still declined. The real data contains no
  easy-negative population at all.
- **We are not modeling topical relevance** (production already handles that);
  we are modeling the residual *"will these two actually connect."* Hence
  absolute AUCs around 0.58–0.64, near-identical query↔match lexical overlap in
  both classes, and TF-IDF keyword cosine plateauing early.

### Latency budget: <100 ms

Accuracy alone isn't a win. A user's query→retrieval round trip must complete
**well within 100 ms**, which constrains the architecture more than the accuracy
target does:

- **Out:** per-candidate LLM calls, cross-encoders scoring each (seeker,
  candidate) pair online, and realistically any remote embedding API on the
  serving path — including `voyage-4-large` itself, whose network round trip
  alone likely eats the budget. It stays the accuracy reference, not necessarily
  a deployable configuration.
- **In:** the **two-tower / bi-encoder shape**, because it factorizes the score
  so candidates are embedded *offline* in batch and the online path is just one
  query encode + an ANN lookup — flat cost as the candidate pool grows. This is
  the main architectural argument for two-tower here, separate from accuracy.
- A merged **LoRA adapter costs nothing extra at serving time** versus frozen
  nano, which makes fine-tuning the cheapest way to buy accuracy under this
  budget.

**Not yet measured** — every number in this repo is offline accuracy; no latency
benchmark exists yet. Standing one up (query encode + ANN retrieve, p50/p95 on
representative hardware) is a prerequisite for declaring any approach a win.

## Approaches

Explore multiple approaches to improve on that holdout number, starting with:

- **Two-tower model** trained on their dataset
- Try **Mixture of Experts** architecture
- Possibly try - **Student - Teacher** architecture
- Later: fine-tune embeddings for intro-matching (Voyage FT enterprise-only; large closed; nano open-weight DIY but overfits at ~200 pairs). Prefer hard labels + two-tower first; student–teacher (Voyage-large → smaller tower) optional after frozen baselines plateau.
- Possible extension: **self-supervised two-tower** (Yao et al.) — add a contrastive auxiliary loss between two randomly field-masked views of the same candidate profile, on top of the supervised pos/neg loss. Targets the long-tail problem (most Boardy users appear in only 1-2 labeled pairs) by learning from the much larger pool of *unlabeled* profiles instead of only the ~660 labeled training pairs. Try only if plain LoRA fine-tuning underperforms the decision gate in [docs/two-tower-fine-tune-plan.md](docs/two-tower-fine-tune-plan.md).
- Possible experiment: **synthesize same-seeker contrastive triples**. In the real data only **9 of 129 seekers** have both an accepted *and* a declined intro — 40 triples total, 25 train-side and 15 in holdout. Those are the only *controlled* comparisons available: holding the seeker fixed cancels out profile quality, writing style, and how easy that person's field is to match in, leaving exactly the residual question this project targets — given production approved both candidates, why did one connection happen and the other not? The synth generator currently emits positives and negatives independently, mostly with *different* seekers (only 5 of 91 synth seekers had both), which is why triplet-style losses were ruled out in [docs/two-tower-fine-tune-plan.md](docs/two-tower-fine-tune-plan.md). Nothing forces that shape: generate one profile + `searchQuery`, then **two candidates against that same anchor** — one plausible-accept, one plausible-decline differing on a single axis. Three payoffs: triplet / `MultipleNegativesRankingLoss` becomes available by construction; it reproduces the most *valuable* real-data structure instead of the most *common* one; and it forces the pos/neg candidates to be mutually comparable, a structural guard against the [docs/possible-bugs.md](docs/possible-bugs.md) #4 failure where the separate `generate_pos.md` / `generate_neg.md` prompts drifted in style until the label was guessable from one profile's text alone (99.2% on synth vs 48.7% on real). The 15 holdout triples are too few for a headline metric but make a sharp diagnostic slice: on a seeker where both answers are known, does the model rank the accepted candidate above the declined one?

## Overview

Boardy’s production embeddings are **Voyage `voyage-4-large`** (32k context),
not classic BERT — see [docs/boardy-embedding-model.md](docs/boardy-embedding-model.md).
This repo still includes a frozen `bert-base-uncased` offline control; we will
benchmark against Voyage and against two-tower / MoE / student–teacher variants.

### Baseline snapshot (offline, 2026-07-17)

| Baseline | ROC-AUC | MRR | Top-1 | R@10 |
|----------|---------|-----|-------|------|
| Frozen BERT | 0.47 | 0.09 | 0.02 | 0.18 |
| Voyage-4-nano (local) | 0.56 | 0.30 | 0.16 | 0.60 |
| Voyage-4-large (API) | 0.57 | 0.31 | 0.13 | 0.70 |

**Large ≈ nano** on this dataset (shared Voyage-4 space); both far above BERT.
Details in the docs page above.

## Data prep

Dedupe pair datasets into a unique-user catalog (canonical `userContactFile`
by highest `userContactFileVersion`):

```bash
python scripts/build_unique_users.py --data-dir data
# writes data/unique_users.json
```

Browse the unique-user catalog in a self-contained HTML page (JSON embedded, no server needed):

```bash
python scripts/build_unique_users_browser.py
# writes data/unique_users_browser.html — open in any browser
```

Per-field BERT token counts (`bert-base-uncased`, `add_special_tokens=False`) plus a filterable/sortable table browser (includes a top summary table: min / mean / median / max per field across all users):

```bash
# use project venv so transformers + cached tokenizer are available
source .venv/bin/activate
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python scripts/build_unique_users_token_counts.py
# writes data/unique_users_token_counts.json
#      and data/unique_users_token_counts_browser.html — open in any browser

# rebuild HTML only (no tokenizer) after template changes:
.venv/bin/python scripts/build_unique_users_token_counts.py --html-only
```

Seed pair summary: [data/dataset_summary.md](data/dataset_summary.md). Plan for growing beyond 200 pairs: [data/synthetic_data_generation.md](data/synthetic_data_generation.md).

### Synthetic pairs (LangGraph + LangSmith)

Pilot pipeline: train-only seed → generate one label → heuristic filter → independent-model judge → staging + batch manifest. See [data/synthetic/pipeline.md](data/synthetic/pipeline.md).

```bash
source .venv/bin/activate
pip install -r requirements.txt

# API key (DeepSeek via OpenRouter — cheap). Copy .env.example → .env
cp .env.example .env
# edit .env and set OPENROUTER_API_KEY=sk-or-v1-...

# user-disjoint seed split (writes data/synthetic/seed_split.json)
python -m synth_pipeline --init-split

# plumbing test (no LLM calls)
python -m synth_pipeline --dry-run --n-pos 2 --n-neg 2 --batch-id dry_demo

# real batch — defaults: deepseek/deepseek-v4-pro (generate),
# google/gemini-3.1-flash-lite (judge) via OpenRouter
# optional LangSmith: LANGCHAIN_TRACING_V2=true LANGCHAIN_API_KEY=...
python -m synth_pipeline --n-pos 10 --n-neg 10 --batch-id batch_001

# after offline human_review.verdict=yes on staged files:
python -m synth_pipeline.promote --batch-id batch_001
```

Outputs land in `artifacts/synth/<batch_id>/` (`staged/`, `dropped/`, `manifest.json`).
Staged envelopes include `metadata.prompt_refs` (hub name + commit when pulled).

#### LangSmith Prompt Hub

Generate/judge system prompts are pulled from LangSmith Hub when configured,
with local `synth_pipeline/prompts/*.md` as fallback (logged on pull failure).

| Env var | Purpose |
|---------|---------|
| `LANGCHAIN_API_KEY` | Hub + tracing (alias: `LANGSMITH_API_KEY`) |
| `LANGSMITH_PROMPT_OWNER` | Handle/org; builds `owner/synth-generate-pos:latest` etc. |
| `LANGSMITH_PROMPT_TAG` | Tag/commit suffix when using owner defaults (`latest`) |
| `SYNTH_PROMPT_GENERATE_POS` | Full id, e.g. `dorby-ai/synth-generate-pos:abc123` |
| `SYNTH_PROMPT_GENERATE_NEG` | Full id for hard-neg generator |
| `SYNTH_PROMPT_JUDGE` | Full id for judge (prefer `:v2` — quality-only schema) |
| `SYNTH_PROMPT_VERSION` | Local fallback label when hub unused (`v1`) |

Judge Hub **v2** returns `would_be_good_intro` only; code computes `judge_verdict`
(pos→pass iff good; neg→pass iff not-good and not easy-neg). Pin with
`SYNTH_PROMPT_JUDGE=<owner>/synth-judge:v2` so generators can stay on
`LANGSMITH_PROMPT_TAG=v1`.

```bash
# 1) Push local markdown into Hub (creates private prompts)
python -m synth_pipeline.push_prompts --owner your-handle --tag v1
# judge polarity fix only:
python -m synth_pipeline.push_prompts --owner your-handle --role judge --tag v2
# dry-run (no API): python -m synth_pipeline.push_prompts --dry-run

# 2) Point the pipeline at Hub (owner shorthand or explicit ids)
# in .env:
#   LANGSMITH_PROMPT_OWNER=your-handle
#   LANGSMITH_PROMPT_TAG=v1
#   SYNTH_PROMPT_JUDGE=your-handle/synth-judge:v2

# 3) Runs without hub still work — local *.md fallback
python -m synth_pipeline --dry-run --n-pos 1 --n-neg 1 --batch-id dry_prompts
```

Hub commit hashes are written into `metadata.prompt_refs` / `prompt_version` and
attached to LangSmith run metadata (`lc_hub_commit_hash`, `prompt_identifier`)
so traces show which prompt version produced each pair.

## Baseline eval (frozen BERT)

Offline bi-encoder baseline: `bert-base-uncased`, mean-pool + L2, cosine
similarity. Runs on Apple MPS when available (else CPU).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# expects data/dataset_positive.json and data/dataset_negative.json
python -m baselines.bert_frozen.eval \
  --data-dir data \
  --model bert-base-uncased \
  --batch-size 16 \
  --max-length 512
```

Writes embeddings cache + `artifacts/bert_frozen/metrics.json` (pair ROC-AUC /
AP / best-F1, retrieval MRR + NDCG/Precision/Recall@K, intent + neg-hardness
slices). See [docs/baseline-metrics.md](docs/baseline-metrics.md). Full metrics:
[docs/baseline-results-all.md](docs/baseline-results-all.md).

## Baseline eval (Voyage-4-nano)

Local open-weight cousin of Boardy’s API `voyage-4-large`, same Voyage-4 embedding
space. Uses `sentence-transformers` with `encode_query` / `encode_document`,
shared field-tagged text from `baselines.bert_frozen.text`. Default
`max_length=8192` (model supports 32k; capped for MPS memory) and
`truncate_dim=1024` (Boardy large default).

```bash
source .venv/bin/activate
pip install -r requirements.txt

python -m baselines.voyage_nano.eval \
  --data-dir data \
  --model voyageai/voyage-4-nano \
  --batch-size 4 \
  --max-length 8192 \
  --truncate-dim 1024
```

Writes embeddings cache + `artifacts/voyage_nano/metrics.json` (same expanded
metrics as frozen BERT; see [docs/baseline-metrics.md](docs/baseline-metrics.md)).
Full metrics: [docs/baseline-results-all.md](docs/baseline-results-all.md).
First run downloads HF weights.

Note: `voyage-4-nano` remote code needs `transformers>=4.51,<5` (transformers 5.x
currently fails with `config_class` None on load). Cold MPS encode ~40+ min;
re-runs hit `artifacts/voyage_nano/` cache.

## Baseline eval (Voyage-4-large API)

Boardy production model via Voyage API (`voyage-4-large`, `output_dimension=1024`,
`input_type=query` for seekers / `document` for candidates). Requires
`VOYAGE_API_KEY`. Per-text disk cache under `artifacts/voyage_large/` so re-runs
cost ~0 tokens. Free tier: first **200M tokens**/account for voyage-4-large;
throttle defaults leave headroom under 3M TPM / 2k RPM.

```bash
source .venv/bin/activate
pip install -r requirements.txt
export VOYAGE_API_KEY=pa-...   # https://docs.voyageai.com/docs/api-key-and-installation

python -m baselines.voyage_large.eval \
  --data-dir data \
  --model voyage-4-large \
  --output-dimension 1024
```

Optional: `--batch-size 16`, `--tpm-limit 2500000`, `--rpm-limit 1500` (or env
`VOYAGE_TPM_LIMIT` / `VOYAGE_RPM_LIMIT`). Writes `artifacts/voyage_large/metrics.json`
+ per-embedding cache (same expanded metrics as BERT/nano; see
[docs/baseline-metrics.md](docs/baseline-metrics.md)). Full metrics:
[docs/baseline-results-all.md](docs/baseline-results-all.md).

## Baseline eval (no-query ablation)

Same three encoders and metrics protocol, but **seeker text drops
`searchQuery`** — only the tagged user profile (`profile_to_text`). Candidate
side is unchanged (match profile only). Purpose: measure how much the query
string contributes vs profile–profile similarity alone.

Shared packing: `baselines/text_no_query.py`. Artifacts go to separate dirs so
caches do not collide with the with-query runs.

```bash
# Frozen BERT (no query)
python -m baselines.bert_frozen_no_query.eval \
  --data-dir data \
  --model bert-base-uncased \
  --batch-size 16 \
  --max-length 512
# → artifacts/bert_frozen_no_query/metrics.json

# Voyage-4-nano (no query)
python -m baselines.voyage_nano_no_query.eval \
  --data-dir data \
  --model voyageai/voyage-4-nano \
  --batch-size 4 \
  --max-length 8192 \
  --truncate-dim 1024
# → artifacts/voyage_nano_no_query/metrics.json

# Voyage-4-large API (no query; needs VOYAGE_API_KEY)
python -m baselines.voyage_large_no_query.eval \
  --data-dir data \
  --model voyage-4-large \
  --output-dimension 1024
# → artifacts/voyage_large_no_query/metrics.json
```

Compare against the matching with-query `artifacts/{bert_frozen,voyage_nano,voyage_large}/metrics.json`.
`scripts/export_baseline_results.py` still aggregates the original three only;
export no-query metrics separately after you run them.
