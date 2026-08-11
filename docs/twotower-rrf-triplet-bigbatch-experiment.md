# Two-tower triplet fine-tune, isolated re-run: bigger batch + multi-negative

Separate experiment from `docs/twotower-rrf-triplet-experiment.md`
(`rrf_triplet_voyage_nano_001` / `rrf_triplet_qwen3_8b_h100_002`). Same
`rrf_003` LLM-judge-labeled synthetic training data, same real 69-pair
holdout for evaluation, **new isolated package**
(`twotower_rrf_triplet_bigbatch/`) — no file under `twotower/` or
`twotower_rrf_triplet/` was touched to build this.

## Why

`rrf_triplet_voyage_nano_001` beat frozen Voyage-4-large on pair AUC (0.610
vs 0.609) but trailed badly on retrieval (recall@1 0.276 vs 0.345, MRR 0.463
vs 0.529). Diagnosis: `MultipleNegativesRankingLoss` only ever contrasted
each anchor against the one negative in its row — and that run's real
per-step batch size was only **2** (`train_batch_size=2`,
`gradient_accumulation_steps=4` gave an *effective* batch of 8, but gradient
accumulation does not add in-batch negatives — only true batch size does).
The model got good at "beat this one specific rival" without practicing
"beat a whole crowd," which is what recall@1 actually requires.

This run changes two things at once, by explicit choice:
1. **Real per-step batch size**, empirically found (see below).
2. **Multiple negatives per anchor per row** (`negative_1`, `negative_2`),
   instead of one.

Both target the same mechanism — more competitors per training step — so
this is **not a single-variable test**; a follow-up ablation would be needed
to attribute the result to one or the other.

## Batch-size probe: a real finding, not what was expected

`voyage-4-nano` is small (347M params) and the first run used batch size 2
with no memory figures ever logged, so there was no existing ceiling to
plan against. An empirical probe (`twotower_rrf_triplet_bigbatch/probe_batch_size.py`
— one real forward+backward step, realistic-length text from the actual
`rrf_003` rows, not placeholder strings) on an 80GB A100 found:

| batch size | result | peak memory |
|---|---|---|
| 8 | OK | 63.98 GB / 79.25 GB (81%) |
| 10 | OOM | — |
| 16 | OOM | — |

The ceiling is a hard **8**, not the 32-64 originally hoped for. Root cause:
at this experiment's real sequence lengths (median 1218 / p95 1509 / max
1701 tokens per text, 4 texts per row with `negatives_per_anchor=2`) and with
**no gradient-checkpointing support for this model**
(`twotower/config.py` already notes "voyage-4-nano remote code currently
rejects gradient checkpointing"), per-example activation memory is large and
scales roughly linearly with batch size — there's no quadratic-attention
trick to exploit here, just a hard activation-memory wall. The actual
training run used **`train_batch_size=6`** (25% safety margin below the
ceiling, for the Trainer's optimizer state / eval pass / step-to-step
variance), `gradient_accumulation_steps=2` (effective batch 12 — free
smoothing, since accumulation doesn't add memory cost, only in-batch
negatives don't benefit from it).

Still a genuine **3x increase in true batch size** (2 → 6) — modest compared
to the original hope, but real.

## Multi-negative extraction: `k=2`, chosen from real coverage

`scripts/build_rrf_multineg_triplets.py --stats-only` on `rrf_003`'s 385
both-class query_keys:

| k | keys with ≥k unique negatives | padding rate at k |
|---|---|---|
| 1 | 385/385 (100%) | 0% |
| 2 | 189/385 (49%) | 51% |
| 3 | 76/385 (20%) | 80% |
| 4 | 28/385 (7%) | 93% |

`k=2` is the largest value where padding (repeating a row's own single real
negative to fill the second slot) stays roughly balanced with genuine
coverage. `k=3`+ would mean 80%+ of rows carry at least one duplicate
negative, diluting rather than strengthening the signal. Final extracted
dataset: **643 rows** (one per positive, not one per pos×neg combination
like the original cartesian extraction), 297 seekers, 27.5% of the 1,286
total negative slots padded.

## Training

- `train_batch_size=6`, `eval_batch_size=6`, `gradient_accumulation_steps=2`,
  `learning_rate=2e-4`, 5 epochs, 245 steps total, ~8.6 min wall-clock on an
  80GB A100.
- Loss fell from 1.40 → 0.84 over training (noisier than the original run's
  curve, expected: the loss now averages over `positive` vs 2 negatives
  instead of vs 1, plus indirect in-batch negatives from 5 other rows per
  step, so the scale and variance of the number itself differs — not
  directly comparable to `rrf_triplet_voyage_nano_001`'s loss values).
- Full loss curve: `artifacts/twotower_rrf_triplet_bigbatch/rrf_triplet_voyage_nano_bigbatch_001/loss_history.json`.
  Raw Modal training log: `training_log.txt` in the same directory.

**Bug found during this run, verified harmless:** checkpoint selection
(`select_best_checkpoint`, reused unmodified from `twotower.train`) keys off
`cfg.primary_metric`, which defaults to `"pair_auc"` — but this experiment's
evaluator (`MultiNegTripletDevEvaluator`) writes `"beat_all_accuracy"`. The
original `twotower_rrf_triplet` package overrode this via
`dataclasses.replace(cfg, primary_metric=...)` right before calling
selection; that line was missed when this package was built, so selection
found no matching metric (`best_score: -inf`) and safely fell back to the
final epoch — the same failure mode already documented in
`docs/possible-bugs.md` #2, now reproduced in a new package. **Checked
whether it mattered**: per-epoch `train_dev_beat_all_accuracy` was 0.433
(ep1) → 0.450 (ep2) → **0.500 (ep3)** → 0.483 (ep4) → **0.500 (ep5)** — epoch
5 ties epoch 3 for the best score, so the shipped final-epoch model is not
worse than what correct selection would have picked. The bug is fixed in
`twotower_rrf_triplet_bigbatch/train.py` for any future rerun, but this run's
numbers stand as reported.

## Results on the real 69-pair holdout

| Metric | This run (bigbatch, k=2) | `rrf_triplet_voyage_nano_001` (batch=2, k=1) | Voyage-4-large (frozen prod) |
|---|---|---|---|
| Pair AUC | 0.576 | **0.610** | 0.609 |
| Hard-negative AUC | 0.588 | **0.614** | 0.602 |
| Retrieval MRR | **0.514** | 0.463 | 0.529 |
| Recall@1 | **0.345** | 0.276 | **0.345** |
| Recall@10 | **0.793** | 0.759 | 0.862 |

**The headline: recall@1 exactly matches Voyage-4-large's frozen baseline**
(0.345, both), and MRR (0.514) closed most of the gap to Voyage-4-large's
0.529 — a real, substantial retrieval improvement over
`rrf_triplet_voyage_nano_001`'s 0.276 / 0.463. This is consistent with the
mechanism this experiment set out to test: more competitors per training
step (bigger batch, 2 negatives instead of 1) taught the model to rank
against a wider field, not just beat one rival.

**But it's a trade, not a strict win**: pair AUC dropped to 0.576, now
*below* both the original triplet run (0.610) and Voyage-4-large (0.609) —
and even below plain TF-IDF's 0.592 (`docs/possible-bugs.md` #3). The
model that got better at "rank against everyone" got worse at the narrow
head-to-head judgment call pair AUC measures. This mirrors, in reverse, the
exact trade-off `docs/twotower-rrf-triplet-experiment.md` §4 (the published
findings report) already described for the *first* run — evidence the two
metrics really do pull in different directions depending on what the
training signal emphasizes, not an artifact of one particular run.

## What this does and doesn't establish

- Confirms the mechanism hypothesis: widening the training-time comparison
  (batch size + negatives per anchor) measurably moves retrieval metrics
  toward Voyage-4-large's, in the direction predicted.
- Does **not** isolate which of the two changed variables (batch size vs.
  negatives-per-anchor) is doing the work — that needs a follow-up run
  holding one fixed.
- Does **not** produce a strictly better model than
  `rrf_triplet_voyage_nano_001` — it trades pair AUC for retrieval. Which
  matters more depends on the product surface (a single committed
  recommendation favors the original run's pair AUC; a ranked shortlist
  favors this run's recall/MRR).
- Same caveats as every other run in this project: `rrf_003` labels are an
  LLM judge's opinion on synthetic profiles, not real accept/decline
  outcomes; the holdout is only 69 pairs / 29 positive queries, so
  individual metric swings of a few points are within noise.

## Reproduce

```bash
# 1. pick k from real coverage (already run; k=2 chosen)
python -m scripts.build_rrf_multineg_triplets --batch-dir exports/rrf_datasets/rrf_003 --stats-only
python -m scripts.build_rrf_multineg_triplets --batch-dir exports/rrf_datasets/rrf_003 \
    --negatives-per-anchor 2 --out artifacts/twotower_rrf_triplet_bigbatch/rrf_003_multineg_k2.json

# 2. empirical batch-size probe (already run; ceiling=8, trained at 6)
modal run twotower_rrf_triplet_bigbatch/probe_batch_size.py --batch-sizes 8,10,12,14

# 3. train (always --detach for multi-hour-capable jobs; this one took ~9 min)
modal run --detach twotower_rrf_triplet_bigbatch/modal_train.py \
    --run-id <new_run_id> --rows-filename rrf_003_multineg_k2.json \
    --negatives-per-anchor 2 --train-batch-size 6 --eval-batch-size 6 \
    --gradient-accumulation-steps 2 --epochs 5
modal app logs <app-id>   # fetch mode, not -f

modal volume get dorby-twotower-rrf-triplet-bigbatch-checkpoints <run_id> \
    ./artifacts/twotower_rrf_triplet_bigbatch/<run_id>
```

## Suggestions for a follow-up

1. **Ablate the two variables.** Run batch=6/k=1 and batch=2/k=2 separately
   against this run's batch=6/k=2 to see which lever drove the recall gain.
2. **A blended objective.** Since pair AUC and recall pull apart, a loss or
   checkpoint-selection metric that blends `beat_all_accuracy` with a
   pairwise term (rather than optimizing purely for one) might land between
   the two runs' trade-off rather than at either extreme.
3. **Revisit `negatives_per_anchor` coverage.** Only 49% of query_keys have
   2 genuinely distinct negatives — a larger or differently-labeled
   `rrf_00N` batch with denser negative coverage per seeker would let `k=3`
   or higher be tried without the padding problem seen here.
4. **This run's checkpoint-selection bug is fixed** in
   `twotower_rrf_triplet_bigbatch/train.py` — any future rerun of this
   package will select the true best epoch by `beat_all_accuracy`, not fall
   back to the final epoch by coincidence.
