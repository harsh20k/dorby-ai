# Two-tower triplet fine-tune on `rrf_003` (in progress)

A separate experimental arm from `twotower/`'s Arm A/B/C (real-data,
pairwise `ContrastiveLoss`) pipeline — kept in its own package
(`twotower_rrf_triplet/`), own artifacts directory
(`artifacts/twotower_rrf_triplet/`), own Modal app/checkpoint volume, so
Arm A/B/C stay exactly reproducible while this runs alongside it. See
[`two-tower-fine-tune-plan.md`](two-tower-fine-tune-plan.md) and
[`twotower-run-001-findings.md`](twotower-run-001-findings.md) for why the
original pipeline used a pairwise loss in the first place (too few seekers
had both a positive and a negative candidate to form real triplets).

## Why now

[`rrf-pairing-pipeline.md`](rrf-pairing-pipeline.md) documents the
"trainability probes" that made this viable: `rrf_002` found 28 of 40
seekers carried both classes (249 triplets) — `run_001`'s pool had 5 of 91.
`rrf_003` (971 profiles, 2619 judged pairs) is the first batch big enough
to build a triplet set worth training on.

## Data provenance — read before trusting any number below

`rrf_003`'s 1175 pos / 1444 neg labels come from an LLM judge
(`google/gemini-3.1-flash-lite`, `naive` framing) grading synthetic
profiles retrieved via dense+BM25 RRF — **not real human accept/decline
outcomes.** `exports/rrf_datasets/rrf_003/manifest.json`'s own
`promotion_note` says so explicitly, and nothing here is or will be
promoted into `data/dataset_*.json`. The judge's own holdout numbers
(`manifest.json`): pair AUC 0.6358, hard-neg AUC 0.6466, decision accuracy
0.5942 — treat any train/dev metric from this experiment as bounded by
"how good the judge's opinion is," not by real matching quality. **The only
number that means anything outside this experiment is the final real
69-pair holdout eval**, run via the unmodified `twotower.eval` /
`baselines/metrics.py` path, exactly like Arm A/B/C.

## Triplet extraction

`scripts/build_rrf_triplets.py` groups `rrf_003`'s judged pairs by
`query_key` (one seeker + one `lookingFor`-derived query — the unit that
determines anchor text, since `seeker_text` depends on both the profile and
the `searchQuery`). For every `query_key` with at least one `pos` and one
`neg`, it emits one triplet per (pos, neg) combination:

```bash
python scripts/build_rrf_triplets.py --batch-dir exports/rrf_datasets/rrf_003 \
    --out artifacts/twotower_rrf_triplet/rrf_003_triplets.json
```

Result: **1,059 query_keys, 385 with both classes → 1,056 triplets across
297 distinct seekers.**

## Architecture

Same base models, same LoRA recipe (`q/k/v/o_proj`, one shared-weight model
called via `encode_query`/`encode_document`) as Arm A/B/C — only the loss
and the data shape change. This run is model-agnostic by design and covers
**two backbones**, via `twotower_rrf_triplet/config.py::MODEL_PRESETS`:

| | `voyage-4-nano` | `Qwen/Qwen3-Embedding-8B` |
|---|---|---|
| Layers (LoRA target count check) | 12 | 36 |
| GPU | Modal L4 | Modal A100-40GB (A10G/24GB OOMs on an 8B model per `docs/hf-embedding-baseline-findings.md`) |
| Train/eval batch size | 2 / 4 | 1 / 1 (grad-accum 8) |
| Query prompt | `"Represent the query for retrieving supporting documents: "` (Voyage convention) | `"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"` — pulled directly from Qwen3-Embedding-8B's own `config_sentence_transformers.json`, not guessed |
| Document prompt | `"Represent the document for retrieval: "` | `""` (Qwen3 ships no document-side instruction) |

**Loss:** `MultipleNegativesRankingLoss` on `(anchor, positive, negative)`
triplets — the one `docs/two-tower-fine-tune-plan.md` originally named as
the target once real triplets existed, replacing `run_001`/Arm A's pairwise
`ContrastiveLoss`. Reuses `twotower.train`'s generic, already-correct
helpers (`build_model`, `add_lora_adapter`, `smoke_backward`,
`select_best_checkpoint`, `collect_env_metadata`) unmodified — nothing in
`twotower/` was edited for this experiment.

**Checkpoint selection:** a new `TripletDevEvaluator`
(`twotower_rrf_triplet/eval_dev.py`) computes `mean(cos(anchor, positive) >
cos(anchor, negative))` on a seeker-disjoint dev slice carved from the
triplets (10% of seekers, same disjointness convention as
`twotower.data.carve_train_dev`, reimplemented locally to avoid any import
coupling to the real-data split path). It writes the same
`train_dev_metrics_epoch{E}_steps{S}.json` file layout
`twotower.train.select_best_checkpoint` already expects, so that unmodified
selection logic works on this experiment's checkpoints too.

**Final evaluation:** always the real 69-pair holdout
(`twotower.data.build_split_bundle` + `twotower.eval.evaluate_pairs`,
unmodified), regardless of which backbone or preset was trained — this is
what gets compared against Arm A / Voyage-4-large / TF-IDF in
[`baseline-results-holdout.md`](baseline-results-holdout.md).

## Compatibility note: Qwen3-Embedding-8B via `encode_query`/`encode_document`

`twotower/eval.py::encode_role` calls `model.encode_query()` /
`model.encode_document()` — confirmed these are **generic
sentence-transformers ≥3.x methods** (present for any model, not a
Voyage-specific remote-code addition), that apply whatever prompts a model
registers under those names. Since Qwen3-Embedding-8B ships a `"query"`
prompt and an empty `"document"` prompt in its own
`config_sentence_transformers.json`, the existing eval path needed **zero
changes** to support it — confirmed by fetching that file directly from
the HF repo rather than assuming.

## Reproduce

```bash
python scripts/build_rrf_triplets.py --batch-dir exports/rrf_datasets/rrf_003

# local dry-run (plumbing only, no backward pass)
python -m twotower_rrf_triplet.train --preset voyage-4-nano --run-id <id> --dry-run

# Modal GPU, real training
modal run twotower_rrf_triplet/modal_train.py --preset voyage-4-nano --run-id rrf_triplet_voyage_nano_001 --epochs 5
modal run twotower_rrf_triplet/modal_train.py --preset qwen3-8b --run-id rrf_triplet_qwen3_8b_001 --epochs 5
modal volume get dorby-twotower-rrf-triplet-checkpoints <run-id> ./artifacts/twotower_rrf_triplet/<run-id>
```

## Status

Plumbing validated: local `--dry-run` (MPS) and a remote Modal `--dry-run`
both complete cleanly for `voyage-4-nano`; a short real (1-epoch) Modal run
is validating the full train → checkpoint-select → holdout-eval path before
committing to full 5-epoch runs on both backbones. **No comparison numbers
yet** — this section will be updated with `voyage-4-nano` and
`Qwen3-Embedding-8B` triplet-arm results on the real holdout once full runs
complete, compared against Arm A (0.579 pair AUC / 0.500 hard-neg AUC) and
Voyage-4-large (0.609 / 0.602).

## What not to conclude from this experiment

- A high train-dev triplet accuracy here means the model agrees with the
  LLM judge on synthetic profiles — it does not mean the model matches real
  human accept/decline behavior. Same caveat that applied to `run_001`'s
  misleading 0.986 train-dev AUC.
- Any win must clear Arm A's real-holdout bar, not just beat `run_001` or
  beat the judge's own holdout AUC.
