# Training without the search query at all

## Question

Every two-tower experiment in this project — `run_001`, the RRF triplet
series, `twotower_qwen_bigbatch`, `top1_ctrl` — trained on seeker text built
as `profile + query` concatenated (`twotower.data.LabeledPair.seeker_text`).
Separately, `twotower_query_weighted/` found that swapping `top1_ctrl`'s
seeker representation to query-only or a query-weighted blend **at eval time
only, with no retraining**, roughly doubles recall@1 (0.19 → 0.29–0.32) on
all 200 real pairs.

That leaves an open question: is the eval-time trick working because the
model under-uses a query it saw plenty of during training, or would training
on profile-only text from the start do just as well, or better? This
experiment trains `top1_ctrl`'s exact recipe on seeker text with **no query
anywhere in training** and checks.

## Method

New isolated package `twotower_no_query/`. Training data:
`scripts/build_rrf_multineg_triplets_no_query.py`, a deliberate near-copy of
`scripts/build_rrf_multineg_triplets.py` with exactly one line changed —
`profile_to_text` instead of `seeker_to_text` for the anchor. Verified
row-for-row against the original `rrf_003_multineg_k1.json`: all 643 rows
differ only on `anchor` text; `seeker_id`/`positive_id`/`negative_ids`/
`positive`/`negatives` are byte-identical, same seed(42)/same 297 seekers/same
0% padding.

`config.py`/`train.py` reproduce `top1_ctrl_001`'s exact launch config
(`artifacts/twotower_top1_optimised/top1_ctrl_001/run_meta.json`):
library-default `MultipleNegativesRankingLoss` (scale=20.0, no hardness
weighting — `top1_ctrl` was the *control* arm of that package), micro-batch 6
/ accum 2 (effective batch 12 → 245 optimizer steps), lr 2e-4, 5 epochs,
`primary_metric="recall@1"` via the same `CorpusRecallDevEvaluator`. A local
`--dry-run` before any GPU spend confirmed an identical 583/60 train/dev split
and identical LoRA target counts (12 per module) to `top1_ctrl`.

Nothing under `twotower/`, `twotower_top1_optimised/`, or
`twotower_query_weighted/` is modified.

```bash
modal run --detach twotower_no_query/modal_train.py --run-id no_query_001
modal volume get dorby-twotower-no-query-checkpoints no_query_001 \
    ./artifacts/twotower_no_query/no_query_001

modal run twotower_no_query/modal_eval.py --run-id no_query_001
modal volume get dorby-twotower-no-query-eval-results no_query_001 \
    ./artifacts/twotower_no_query/no_query_001_real200
```

Cost: ~6 minutes on A100-80GB for training (245 steps), plus a small L4
inference job for the all-200 eval — well under $2 total.

## Results — all 200 real pairs

| Seeker representation | Pair AUC | Hard-neg AUC | MRR | Recall@1 | Recall@10 |
|---|---|---|---|---|---|
| `top1_ctrl`, trained **with** query (concat, published) | 0.5683 | 0.5484 | 0.3550 | 0.19 | 0.69 |
| `top1_ctrl` + eval-time **query_only** swap (no retrain) | 0.5945 | 0.6456 | **0.5076** | **0.32** | **0.90** |
| `top1_ctrl` + eval-time **alpha_0.6** blend (no retrain) | **0.6129** | 0.6196 | 0.4818 | 0.29 | 0.87 |
| **`no_query_001`, trained without query at all** | 0.5718 | 0.5550 | 0.3371 | 0.18 | 0.67 |
| Frozen, untrained voyage-4-nano (reference) | — | — | — | 0.18 | — |

**Training without the query does not replicate the eval-time trick — it is
the weakest of the four on recall@1 and MRR, landing almost exactly at the
untrained frozen baseline (R@1 0.18, identical to the untrained reference).**
It is also slightly *worse* than `top1_ctrl`'s own concat-trained numbers on
MRR (0.3371 vs 0.3550) and about the same on pair AUC and hard-negative AUC.
The LoRA fine-tune essentially adds nothing when it never sees a query during
training.

**The 69-pair holdout misled again — for the fourth+ time in this project.**
On the holdout alone, `no_query_001` looked like the best fine-tune ever
produced here: recall@1 **0.4138**, MRR **0.5574**, both higher than
`top1_ctrl`'s own holdout numbers. Scored on all 200, that reverses — it's
the worst recall@1 of the four rows above. This is the same holdout-vs-all-200
divergence documented in `docs/all-200-baseline-sweep.md` and
`docs/twotower-qwen-bigbatch-experiment.md`; the standing project rule
("score on all 200 before calling anything neutral") held again.

## What this means

The eval-time query-weighting win on `top1_ctrl` is **not** explained by "the
model wasn't trained to lean on the query enough." A model that never saw the
query at all does not recover — or even approach — that win; it just
regresses to the untrained baseline. Whatever the eval-time trick is
exploiting (query text carrying a cleaner, less-diluted signal than the
profile — see `docs/query-weighted-encoding-experiment.md`'s Jaccard-overlap
finding) seems to depend on the model having learned *something* from the
query during training, even though `top1_ctrl` itself only saw it concatenated
and never separated. Training a model with genuinely separated
query/profile encoders (rather than one shared string) is a plausible next
step, but is out of scope for this experiment.

## Reproduce

```bash
python scripts/build_rrf_multineg_triplets_no_query.py \
    --batch-dir exports/rrf_datasets/rrf_003 --negatives-per-anchor 1 \
    --out artifacts/twotower_no_query/rrf_003_multineg_k1_no_query.json

modal run --detach twotower_no_query/modal_train.py --run-id no_query_001
modal volume get dorby-twotower-no-query-checkpoints no_query_001 \
    ./artifacts/twotower_no_query/no_query_001

modal run twotower_no_query/modal_eval.py --run-id no_query_001
modal volume get dorby-twotower-no-query-eval-results no_query_001 \
    ./artifacts/twotower_no_query/no_query_001_real200
```
