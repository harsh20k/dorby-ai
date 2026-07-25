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
found that **Qwen3-Embedding-8B (Apache 2.0, free) beats Voyage-4-large —
Boardy's own production model — on the core accept/decline task** (pair
ROC-AUC 0.6595 vs 0.6086), the first model of any kind in this project to
do so. Full results and two real model-loading compatibility issues found
and fixed along the way in `docs/hf-embedding-baseline-findings.md`.

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
which is a documented approximation.

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
  --out docs/pairs-comparison-graph.html

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
