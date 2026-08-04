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

modal run twotower_no_query/modal_eval_matched.py --run-id no_query_001
modal volume get dorby-twotower-no-query-eval-results no_query_001_matched \
    ./artifacts/twotower_no_query/no_query_001_matched
```

Cost: ~6 minutes on A100-80GB for training (245 steps), plus a small L4
inference job for the all-200 eval — well under $2 total.

## Bug found: the first eval fed the model text it never trained on

The first pass at this evaluation reused `eval_real_full.eval.run_eval`
(→ `twotower.eval.evaluate_pairs`) exactly like every other adapter in this
project. That function builds seeker text via
`twotower.data.LabeledPair.seeker_text`, which is **hardcoded** to
`profile + query` concatenated — correct for `top1_ctrl` and every other
adapter here, all of which trained on that same text. It is wrong for
`no_query_001`, which trained on `profile_to_text` alone: the eval silently
fed it query tokens it had never seen during training. A user catch (not a
self-caught bug) flagged this before the result was taken as final —
documented in full as `docs/possible-bugs.md` #5.

Re-scored on seeker text that actually matches training (`profile_to_text`,
no query), reusing `twotower_query_weighted.eval`'s already-published
`profile_only` scoring path read-only:

| Eval text | Pair AUC | Hard-neg AUC | MRR | Recall@1 | Recall@10 |
|---|---|---|---|---|---|
| Mismatched (query included — the bug) | 0.5718 | 0.5550 | 0.3371 | 0.18 | 0.67 |
| **Matched (profile only — correct)** | 0.5574 | 0.5374 | 0.2827 | **0.13** | 0.62 |

The corrected number is *lower*, not higher — training without the query
performs even more modestly than the first (buggy) pass suggested. All
results below use the corrected, matched-distribution number.

## Results — all 200 real pairs

| Seeker representation | Pair AUC | Hard-neg AUC | MRR | Recall@1 | Recall@10 |
|---|---|---|---|---|---|
| `top1_ctrl`, trained **with** query (concat, published) | 0.5683 | 0.5484 | 0.3550 | 0.19 | 0.69 |
| `top1_ctrl` + eval-time **query_only** swap (no retrain) | 0.5945 | 0.6456 | **0.5076** | **0.32** | **0.90** |
| `top1_ctrl` + eval-time **alpha_0.6** blend (no retrain) | **0.6129** | 0.6196 | 0.4818 | 0.29 | 0.87 |
| `top1_ctrl` + eval-time **profile_only** swap (no retrain) | 0.5489 | 0.5272 | 0.2800 | 0.13 | 0.59 |
| **`no_query_001`, trained without query, scored matched** | 0.5574 | 0.5374 | 0.2827 | 0.13 | 0.62 |

**The real finding: whether the query is present during training makes almost
no difference to how the model performs on profile-only input.**
`top1_ctrl` (trained with the query, then evaluated on profile-only text at
eval time) and `no_query_001` (trained *without* the query, evaluated on the
same profile-only text) land within noise of each other on every metric —
identical recall@1 (0.13), nearly identical MRR (0.2800 vs 0.2827) and AUC
(0.5489 vs 0.5574). Seeing the query during training does not measurably
change the model's profile-only representation quality one way or the other.

What *does* move the numbers, dramatically, is what the model is asked to
encode **at eval time** — `top1_ctrl` scored 0.13 R@1 on profile-only text and
0.32 R@1 on query-only text, a 2.5x swing, using the exact same trained
weights. The representation gap is about what's fed in at inference, not
about what the model saw in training.

**The 69-pair holdout misled again — for the fourth+ time in this project.**
On the holdout alone (still using the mismatched eval at the time), the first
pass at `no_query_001` looked like the best fine-tune ever produced here:
recall@1 **0.4138**, MRR **0.5574** — both higher than `top1_ctrl`'s own
holdout numbers. That number was doubly unreliable — mismatched eval text
*and* a small, noisy population — and neither the ranking nor the mismatch
survived scoring on all 200. The standing project rule ("score on all 200
before calling anything neutral") held again, independent of the bug above.

## What this means

The eval-time query-weighting win on `top1_ctrl` is **not** explained by "the
model wasn't trained to lean on the query enough," and it's also not
explained by "training without the query produces a worse profile encoder" —
neither training condition changes profile-only performance measurably.
What matters is what text is fed into the (frozen, already-trained) encoder
at query time: query-only or query-weighted text carries a cleaner, less
diluted signal than the profile does (see
`docs/query-weighted-encoding-experiment.md`'s Jaccard-overlap finding), and
that appears to hold regardless of whether the query was concatenated into
training data or withheld entirely. Training a genuinely separated
query/profile two-tower encoder (rather than one shared string, as this
project's whole `twotower/` family uses) remains a plausible way to test
whether training can improve on the eval-time trick, but this experiment
shows plain query-inclusion-or-not in training is not the lever.

## Reproduce

```bash
python scripts/build_rrf_multineg_triplets_no_query.py \
    --batch-dir exports/rrf_datasets/rrf_003 --negatives-per-anchor 1 \
    --out artifacts/twotower_no_query/rrf_003_multineg_k1_no_query.json

modal run --detach twotower_no_query/modal_train.py --run-id no_query_001
modal volume get dorby-twotower-no-query-checkpoints no_query_001 \
    ./artifacts/twotower_no_query/no_query_001

modal run twotower_no_query/modal_eval_matched.py --run-id no_query_001
modal volume get dorby-twotower-no-query-eval-results no_query_001_matched \
    ./artifacts/twotower_no_query/no_query_001_matched
```
