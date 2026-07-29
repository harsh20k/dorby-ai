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

### `voyage-4-nano`, full 5 epochs — real 69-pair holdout result

**Beats every twotower arm to date, and edges out frozen Voyage-4-large
(Boardy's own production model) on both classification metrics:**

| Metric | This run (rrf_triplet_voyage_nano_001) | Arm A (real-only) | `run_001` | Voyage-4-large (frozen prod) |
|---|---|---|---|---|
| Pair AUC | **0.6103** | 0.579 | 0.578 | 0.609 |
| Hard-negative AUC | **0.6138** | 0.500 (chance) | 0.4845 (below chance) | 0.602 |
| Retrieval MRR | 0.4633 | 0.388 | 0.283 | 0.529 |

Best checkpoint was epoch 4 (dev triplet-accuracy 0.660), selected
automatically by the reused `select_best_checkpoint` safety net. Full
metrics: `artifacts/twotower_rrf_triplet/rrf_triplet_voyage_nano_001/metrics_holdout.json`.

This is the first twotower fine-tune of any kind in this project to beat
Voyage-4-large on pair AUC. Retrieval MRR still trails (0.463 vs 0.529) —
the win is on classification, not precise top-of-list ranking.

### `Qwen/Qwen3-Embedding-8B`, full 5 epochs — real 69-pair holdout result

**New best pair AUC in the entire project — beats every twotower arm, both
frozen Voyage models, and the frozen Qwen3-Embedding-8B baseline itself:**

| Metric | This run (rrf_triplet_qwen3_8b_h100_002) | `voyage-4-nano` (this experiment) | Arm A | Voyage-4-large (frozen prod) | Frozen Qwen3-Embedding-8B baseline |
|---|---|---|---|---|---|
| Pair AUC | **0.6672** | 0.6103 | 0.579 | 0.609 | 0.6595 |
| Hard-negative AUC | **0.6172** | 0.6138 | 0.500 | 0.602 | — |
| Retrieval MRR | 0.4390 | 0.4633 | 0.388 | 0.529 | — |

Ran on H100 after the A100-40GB attempts were killed for unrelated
infrastructure reasons (see "Operational lessons"). Training took ~24
minutes end to end (600 steps, ~1463s per `train_runtime`).

**Important caveat — the reported result is from the final epoch, not the
best one, because `save_total_limit=3` pruned the actual best checkpoint
before selection could use it.** Per-epoch dev triplet-accuracy from
`loss_history.json`:

| Epoch | Dev triplet-accuracy |
|---|---|
| 1 | 0.680 |
| **2** | **0.711 (peak)** |
| 3 | 0.608 (drop) |
| 4 | 0.619 |
| 5 (final, used for holdout) | 0.649 |

Only checkpoints 360/480/600 (epochs 3-5) survived `save_total_limit=3` by
the time `select_best_checkpoint` ran — epoch 2's checkpoint (the actual
best on dev) had already been deleted, so selection fell back to
`final_in_memory` (epoch 5). **The reported 0.6672 pair AUC may
understate what this recipe can do** — a rerun with `save_total_limit=5`
(keep every epoch) could plausibly do even better if the epoch-2 checkpoint
generalizes similarly well to the real holdout. This is the same failure
mode `docs/twotower-run-001-findings.md`'s distillation side-experiment hit.

Full loss curve, dev-accuracy-per-epoch, and raw training logs saved at
`artifacts/twotower_rrf_triplet/rrf_triplet_qwen3_8b_h100_002/` (`loss_history.json`,
`training_log_part1_pre_interrupt.txt`, `training_log_part2_resumed.txt` — see
below for why there are two parts). `voyage-4-nano`'s loss curve (reconstructed
post-hoc from Modal's server-side log retention, since that run predates
`loss_history.json` being written) is at
`artifacts/twotower_rrf_triplet/rrf_triplet_voyage_nano_001/loss_history.json`
+ `training_log.txt`.

## Operational lessons from launching these runs

- **fp32 load of an 8B model OOMs even batch_size=1 on a 40GB A100** (39.4GB
  used, tried to allocate 96MiB more) — fixed via bf16 loading
  (`twotower_rrf_triplet/model.py::build_model_with_dtype`), matching the
  already-proven `baselines/hf_embedding` path for the same model family.
  `select_best_checkpoint`'s reload step needed the same fix (it would have
  re-OOM'd at the very end of a multi-hour run, reloading the checkpoint in
  fp32 again) — `twotower_rrf_triplet/checkpoint.py::select_best_checkpoint_with_dtype`.
- **`modal run` without `--detach` ties the remote job to the local CLI
  process.** Two full runs launched via plain `modal run` both died with
  `RemoteError: Function call was cancelled by user or a failure` when their
  local foreground process was interrupted — `voyage_nano_001` had actually
  finished training and picked its best checkpoint before dying only during
  the final holdout-eval step (recovered for free: the adapter was already
  saved, so holdout eval was just re-run locally against it, no retraining
  needed); `qwen3_8b_002` died too early to produce anything. Always launch
  multi-hour Modal training with `--detach`.
- **Training loss wasn't being persisted** — `train.py` now writes
  `loss_history.json` (per-step training loss + dev-eval points, extracted
  from `trainer.state.log_history`) alongside the existing metrics files.
  `voyage_nano_001` predates this; its loss curve was instead reconstructed
  post-hoc from Modal's server-side log retention (`modal app logs --tail
  5000`) — the locally-captured log only kept the last ~200 lines since it
  was originally piped through `tail`.
- **A genuinely unexplained mid-run cancellation hit `qwen3_8b_h100_002` at
  step 472/600 (79%, epoch 3.92)** — Modal's own log showed `Received a
  cancellation signal while processing input` followed by a forced kill 30s
  later, with no action taken from this session at the time. Recovered with
  zero data loss: `save_strategy="epoch"` had already checkpointed epoch 3
  (`checkpoint-360`) to the volume, so relaunching with
  `--resume-from-checkpoint .../checkpoint-360` picked up exactly where it
  left off — confirmed via the resumed run's log showing progress jump
  straight to step 361. HF's checkpoint mechanism also transparently merged
  the pre- and post-interruption `log_history` into one continuous curve
  (visible in the final `loss_history.json` — no gap or duplication at step
  360), so the saved loss data is a complete, accurate record despite the
  interruption. Root cause of the cancellation itself was never identified.
- **Switching from A100-40GB to H100 mid-flight (user request, for speed)
  cost one wasted attempt** — `qwen3_8b_h100_001` was killed on the
  (wrong) assumption that "pending, no logs" meant it was stuck; it was
  actually running fine. The real lesson: `modal run --detach` exits its
  local watcher process almost immediately after confirming detachment, so
  a background-bash log tail captures nothing after that point — that
  looked identical to a hang. The correct way to check a detached job's
  real status is `modal app logs <app-id>` (fetches recent entries without
  needing `-f`), confirmed working from `qwen3_8b_h100_002` onward.

## Suggestions for improving on this further

- **Increase effective batch size for `MultipleNegativesRankingLoss`.**
  This loss uses every other example's positive in the batch as a free
  extra in-batch negative, so a larger batch is a harder, more informative
  contrastive task, not just faster training. Current effective batch
  (`train_batch_size × gradient_accumulation_steps`) is only 8 for both
  presets — `voyage-4-nano` (347M params) has large headroom on an A100 to
  go much higher (e.g. 32-64) with no accumulation needed.
- **More epochs, gated on `loss_history.json`'s dev-accuracy trend, not a
  blind increase** — 5 epochs was inherited from Arm A's pairwise-loss
  recipe; worth checking whether triplet-accuracy is still climbing at
  epoch 5 before extending.
- **Tune `MultipleNegativesRankingLoss`'s `scale` (temperature)** — default
  20.0, untouched so far; controls how sharply positives are pushed apart
  from in-batch negatives.
- **Cap triplets per query_key** — current cartesian-product extraction
  lets query_keys with several judged pos/neg dominate gradient updates;
  capping negatives sampled per positive would balance this.
- **LoRA rank** — currently 8/16 for both models, inherited from Arm A;
  raise cautiously given `run_001`'s history of the adapter overfitting to
  judge/generation artifacts rather than real signal.
- **Step-based eval/checkpointing** instead of epoch-based, for finer
  checkpoint-selection resolution (Arm A's distillation side-experiment lost
  its best checkpoint to `save_total_limit` pruning before selection saw it
  — same risk here with only 5 epoch-level checkpoints).

## What not to conclude from this experiment

- A high train-dev triplet accuracy here means the model agrees with the
  LLM judge on synthetic profiles — it does not mean the model matches real
  human accept/decline behavior. Same caveat that applied to `run_001`'s
  misleading 0.986 train-dev AUC.
- Any win must clear Arm A's real-holdout bar, not just beat `run_001` or
  beat the judge's own holdout AUC.
