# Training on the search query alone

## Question

Third leg of a three-way seeker-text comparison. `twotower_top1_optimised/
top1_ctrl` trained on profile+query concatenated. `twotower_no_query/` then
trained on profile only and found the result lands within noise of just
swapping `top1_ctrl` to profile-only text at eval time (no retraining) — i.e.
training condition barely moves profile-only performance. This experiment
asks the mirror question for the query side: does actually training on
query-only text beat `top1_ctrl`'s eval-time query-only swap — the best
fine-tuned-model retrieval number on record (recall@1 0.32)?

Prediction stated in advance (per `twotower_no_query/`'s finding): probably
lands close to the eval-time swap, not meaningfully above it.

## Method

New isolated package `twotower_query_only/`, exact `top1_ctrl` recipe (see
`config.py`): library-default `MultipleNegativesRankingLoss`, `recall@1`
checkpoint selection, micro-batch 6/accum 2 (245 optimizer steps), lr 2e-4,
5 epochs, A100-80GB. Training data:
`scripts/build_rrf_multineg_triplets_query_only.py`, reusing
`query_weighted.text.query_only` read-only for the anchor text (its
empty-query-falls-back-to-profile behavior, not reimplemented) — verified
row-for-row identical to the original k1 file on every field except `anchor`
(643/643 rows differ only there; 0 rows needed the empty-query fallback).
Local `--dry-run` confirmed an identical 583/60 split and LoRA target counts
before any GPU spend.

**Evaluation was built matched-distribution from the start this time** —
`twotower_no_query/`'s first pass had a bug (`docs/possible-bugs.md` #5)
where the default eval path fed the model text it never trained on.
`twotower_query_only/modal_eval_matched.py` scores this adapter only on
`query_only` seeker text, via `twotower_query_weighted.eval`'s already-
published `query_only` scoring path — the exact function that scored
`top1_ctrl`'s own query-only eval-time swap, so the two numbers are directly
comparable by construction.

```bash
python scripts/build_rrf_multineg_triplets_query_only.py \
    --batch-dir exports/rrf_datasets/rrf_003 --negatives-per-anchor 1 \
    --out artifacts/twotower_query_only/rrf_003_multineg_k1_query_only.json

modal run --detach twotower_query_only/modal_train.py --run-id query_only_001
modal volume get dorby-twotower-query-only-checkpoints query_only_001 \
    ./artifacts/twotower_query_only/query_only_001

modal run twotower_query_only/modal_eval_matched.py --run-id query_only_001
modal volume get dorby-twotower-query-only-eval-results query_only_001_matched \
    ./artifacts/twotower_query_only/query_only_001_matched
```

Cost: ~6 minutes A100-80GB training + a small L4 inference eval, well under
$2 total.

## Results — all 200 real pairs

| Seeker representation | Pair AUC | Hard-neg AUC | MRR | Recall@1 | Recall@10 |
|---|---|---|---|---|---|
| `top1_ctrl` + eval-time **query_only** swap (no retrain) | 0.5945 | 0.6456 | 0.5076 | **0.32** | 0.90 |
| **`query_only_001`, trained on query only, matched eval** | 0.5952 | **0.6492** | 0.4985 | 0.29 | **0.91** |

**The prediction held.** The two rows are within noise of each other on
every metric — AUC nearly identical (0.5945 vs 0.5952), hard-neg AUC actually
slightly *higher* for the trained model (0.6492 vs 0.6456), MRR and recall@1
slightly lower (0.4985 vs 0.5076, 0.29 vs 0.32), recall@10 slightly higher
(0.91 vs 0.90). No consistent direction of advantage either way — this reads
as two draws from the same underlying performance level, not two genuinely
different models.

## What this means, taken together with `twotower_no_query/`

Both halves of the seeker-text triangle now say the same thing:

| Representation | Trained that way | Swapped at eval time only | Gap |
|---|---|---|---|
| Profile only | 0.13 R@1 (`no_query_001`, matched) | 0.13 R@1 (`top1_ctrl` swap) | ~0 |
| Query only | 0.29 R@1 (`query_only_001`, matched) | 0.32 R@1 (`top1_ctrl` swap) | ~0 |

**Training the model specifically on a representation does not measurably
beat just handing that representation to a normally-trained model at eval
time.** The two-tower LoRA adapter's usefulness comes from what it's asked
to encode *at inference*, not from having specialized on that exact input
shape during training. Practically, this closes the case for training a
specialized query-only or profile-only encoder as a way to chase the
eval-time win further — on this recipe, data, and scale, it isn't a lever.
What *is* a lever, confirmed three times now (frozen nano, frozen
voyage-4-large, and both trained/swapped fine-tune pairs): which text you
choose to encode at query time, full stop. See
`docs/query-weighted-encoding-experiment.md` and
`docs/twotower-no-query-experiment.md`'s voyage-4-large extension for the
size of that lever on its own (voyage-4-large query-only: recall@1 0.42, the
best number anywhere in this project).

Published alongside the `twotower_no_query/` findings in the same artifact:
https://dorby-project-story-411960113601.s3.amazonaws.com/docs/twotower-no-query-comparison.html
