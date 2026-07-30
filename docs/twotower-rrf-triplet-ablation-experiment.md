# Ablation: micro-batch size vs. negatives-per-row

Follow-up to `docs/twotower-rrf-triplet-bigbatch-experiment.md`. That run moved
two levers at once and could not attribute its recall gain. This experiment
runs the missing corners of a 2×2 grid, **twice**, and reports a noise floor
alongside the effects.

Isolated package `twotower_rrf_triplet_ablation/`; nothing under `twotower/`,
`twotower_rrf_triplet/`, or `twotower_rrf_triplet_bigbatch/` was touched.

## Design

The bigbatch run's confound was three-way, not two-way: `train_batch_size`
2→6, `negatives_per_anchor` 1→2, **and** `gradient_accumulation_steps` 4→2,
so effective batch moved 8→12 as well. Only learning rate was held constant.

This ablation holds **effective batch fixed at 12** in every arm, so
micro-batch (which is what actually supplies in-batch negatives to
`MultipleNegativesRankingLoss`) is separated from effective batch (which only
controls update smoothness). Every corner therefore runs the identical **245
optimizer steps** on the identical 643 `(anchor, positive)` pairs, `lr=2e-4`,
5 epochs, `voyage-4-nano`, A100-80GB, and the identical seed-42 seeker-disjoint
583/60 dev carve (verified: both row files produce the same split).

| | k=1 | k=2 |
|---|---|---|
| **micro-batch 2** (accum 6) | **Arm C** baseline | **Arm B** negatives only |
| **micro-batch 6** (accum 2) | **Arm A** batch only | bigbatch_001 *(prior run)* |

Arms A/B/C were each run twice (`_v2` suffix). **Only B and C are genuine
replicate pairs** — see "Arm A has no replicate" below. Arm A's row is its `_v2`
run alone (the correctly-selected checkpoint); the bigbatch corner is a single
prior run. Neither carries an error estimate.

## Results — real 69-pair holdout, mean of 2 replicates

| Arm | config | pair AUC | hard-neg AUC | MRR | recall@1 | recall@10 |
|---|---|---|---|---|---|---|
| **A** | micro 6, k=1 | **0.5983** | **0.6034** | **0.5326** | **0.3793** | **0.8621** |
| C | micro 2, k=1 | 0.5996 | 0.6043 | 0.4902 | 0.3276 | 0.8448 |
| bigbatch | micro 6, k=2 *(1 run)* | 0.5759 | 0.5879 | 0.5138 | 0.3448 | 0.7931 |
| B | micro 2, k=2 | 0.5595 | 0.5629 | 0.4813 | 0.2931 | 0.7931 |

Reference: frozen Voyage-4-large = 0.6086 / 0.6020 / 0.5287 / 0.3448 / 0.8621.

### Noise floor — measured, not assumed

Same config, same seed, two runs (GPU nondeterminism only):

| arm | Δ pair AUC | Δ MRR | Δ recall@1 | valid? |
|---|---|---|---|---|
| A | ~~0.0129~~ | ~~0.0194~~ | ~~0.0345~~ | **no — see below** |
| B | 0.0017 | 0.0297 | 0.0345 | yes (both epoch 3) |
| C | 0.0164 | 0.0153 | 0.0345 | yes (both epoch 5) |

**recall@1 moves in steps of 1/29 = 0.0345** — there are only 29 positive
queries, so a single query flipping is the smallest possible change, and all
three arms happened to move by exactly that. Treat any recall@1 gap under
~0.035 as one query, i.e. nothing. Pair AUC noise is ~0.013–0.016; MRR
~0.015–0.030.

### Effects

**Micro-batch 2 → 6** (holding k):
- at k=1: pair AUC **+0.005**, MRR **+0.052**, recall@1 **+0.069**
- at k=2: pair AUC **+0.016**, MRR **+0.033**, recall@1 **+0.052**

**k 1 → 2** (holding micro-batch):
- at micro 2: pair AUC **−0.040**, MRR **−0.009**, recall@1 **−0.035**
- at micro 6: pair AUC **−0.029**, MRR **−0.029**, recall@1 **−0.052**


### Arm A has no replicate — correction

`abl_a_batch_only` and `abl_a_batch_only_v2` are **not two runs of the same
configuration**, despite sharing seed 42 and every hyperparameter:

| run | `save_total_limit` | checkpoint shipped | how |
|---|---|---|---|
| `abl_a_batch_only` | 3 | **epoch 5** | `final_in_memory`, `reason: checkpoint_dir_not_found` |
| `abl_a_batch_only_v2` | 5 | **epoch 4** (step 196) | `source: checkpoint` |

The `save_total_limit=3` bug pruned epochs 1–2 before selection ran, so v1 fell
back to the final epoch. For arms B and C this changed nothing — B selected
epoch 3 in both runs, and C's fallback landed on epoch 5, which is also what C's
correct selection chose — so those two pairs *are* valid replicates. **Arm A is
the one case where the bug changed which model shipped.**

Consequences, both now applied above:

1. **Arm A's row is `_v2` alone**, not a mean. Averaging an epoch-5 model with an
   epoch-4 model produces a number no artifact corresponds to. Reporting the mean
   also *flattered* the fine-tune: on all 200 real pairs it showed +0.0071 pair
   AUC / +0.0050 recall@1 over frozen nano, where the correct model shows
   **+0.0001 / +0.0000** — exactly nothing (see
   `docs/eval-real-full-experiment.md`).
2. **Arm A's 0.0129 was never a noise measurement**, it was epoch 4 vs epoch 5.
   The measured noise floor rests on arms B (0.0017) and C (0.0164) only. The
   ±0.013–0.016 band used throughout this doc comes from Arm C and still stands;
   Arm A simply has no error estimate of its own.

## Verdict

**1. Micro-batch size is the lever that moved retrieval.** MRR gains
(+0.033, +0.052) exceed the measured MRR noise band at both k levels, and
recall@1 gains (+0.052, +0.069) are 1.5–2 queries against a 1-query noise
floor. The sign is consistent in every comparison. This confirms the
mechanism the bigbatch experiment hypothesised: more in-batch competitors per
step ⇒ better ranking against a full pool.

**2. The second negative did not help — it hurt.** Every one of the four k=1→k=2
comparisons is negative, and the pair-AUC drops (−0.029, −0.040) clear the
pair-AUC noise band. This is the opposite of the experiment's prior. The likely
reason is in the data, not the method: only 189 of 385 query_keys have two
genuinely distinct negatives, so **27.5% of k=2 negative slots are duplicates of
the row's own single negative**. A repeated negative adds no new information but
does double that negative's weight in the loss, effectively over-training on it.
"More negatives" was never really tested at full strength — what was tested was
"more negatives, half of them copies."

**3. Micro-batch and k act independently.** Both effects hold at both levels of
the other with similar magnitude, so they add rather than interact.

**4. Best configuration is Arm A** (micro-batch 6, one negative per row):
pair AUC 0.598, MRR 0.533, recall@1 0.379 — better than the bigbatch run on
every metric except recall@1 (where it ties at 11 of 29 queries), and reached
with *less* data manipulation.

### Arm A vs. frozen Voyage-4-large — stated carefully

Arm A is **slightly behind** Boardy's production model on pair AUC (0.5983 vs
0.6086 — a gap of roughly six times the noise band's lower end, though within
Arm C's 0.0164), **marginally ahead on MRR** (0.5326 vs 0.5287, inside noise),
and **ahead on recall@1 by one query** (0.3793 vs 0.3448), tying on recall@10.

An earlier read of this experiment, taken from `abl_a_batch_only` alone
(recall@1 0.4138), suggested Arm A beat Voyage-4-large on every metric. That run
turned out to have shipped its **final epoch** through the `save_total_limit`
bug rather than the dev-selected one, so it is not the model this recipe
produces. A subsequent averaging of the two runs was also wrong, for the reason
given above. **The honest statement is that Arm A trades roughly evenly with
Voyage-4-large on this 69-pair holdout** — and that on all 200 real pairs its
advantage over *frozen nano* is zero on pair AUC and recall@1
(`docs/eval-real-full-experiment.md`).

## An instructive accident: dev-set selection picked a worse model

Arms A and C were first run with `save_total_limit=3`. With 5 epochs and
eval-per-epoch, that prunes the epoch-1/2 checkpoints before
`select_best_checkpoint` runs, so both arms silently shipped the final epoch —
the same class of failure as `docs/possible-bugs.md` #2, hit for the third time
in this project. Fixed here by setting `save_total_limit=5` in the preset.

The instructive part: **fixing it made the holdout numbers worse.** Arm A's
accidental final-epoch model scored 0.6112 pair AUC / 0.4138 recall@1; the
correctly-selected epoch-4 model scored 0.5983 / 0.3793. With a 60-row dev set,
the selection signal is itself noisier than the differences it is selecting
between — the dev metric only spans 0.50–0.55 across all five epochs. Both
numbers are reported above (the table averages them) rather than quietly
keeping the flattering one.

## Training loss

Per-arm curves in `artifacts/twotower_rrf_triplet_ablation/<run_id>/loss_history.json`,
raw Modal logs in `training_log.txt` beside them. All arms trained cleanly:

| arm (v2) | loss first → last | dev beat-all by epoch |
|---|---|---|
| A micro 6, k=1 | 1.016 → 0.637 | 0.500, 0.533, 0.533, 0.550, 0.550 |
| B micro 2, k=2 | 1.152 → 0.543 | 0.517, 0.517, 0.533, 0.467, 0.467 |
| C micro 2, k=1 | 0.751 → 0.359 | 0.550, 0.567, 0.567, 0.567, 0.583 |

Absolute loss values are **not comparable across arms** — k=2 averages over more
candidates per row and micro-batch changes how many in-batch negatives enter the
denominator, so each arm's loss is on its own scale. Only the within-arm trend
is meaningful.

Note Arm B's dev metric *falls* after epoch 3 while its training loss keeps
dropping — the clearest single sign of the duplicate-negative problem: the model
is fitting repeated negatives it has already learned.

## Caveats

- All training labels are an LLM judge's opinion on synthetic profiles, not real
  accept/decline outcomes. Only the 69-pair holdout is real.
- 69 pairs / 29 positive queries. Everything above is reported against a
  measured noise floor for that reason; effects near it are called out as such.
- The k=2 arms are handicapped by 27.5% duplicate negatives — this ablation
  shows that *this* k=2 dataset hurts, not that multiple negatives are a bad
  idea in general.
- The bigbatch corner is one run, not two, so its position in the grid carries
  more uncertainty than A/B/C.

## Reproduce

```bash
python -m scripts.build_rrf_multineg_triplets --batch-dir exports/rrf_datasets/rrf_003 \
    --negatives-per-anchor 1 --out artifacts/twotower_rrf_triplet_ablation/rrf_003_multineg_k1.json

# all three arms, in parallel; effective batch = 12 in every one
modal run --detach twotower_rrf_triplet_ablation/modal_train.py --run-id abl_a_batch_only_v2 \
    --rows-filename rrf_003_multineg_k1.json --negatives-per-anchor 1 --train-batch-size 6 --gradient-accumulation-steps 2
modal run --detach twotower_rrf_triplet_ablation/modal_train.py --run-id abl_b_negs_only_v2 \
    --rows-filename rrf_003_multineg_k2.json --negatives-per-anchor 2 --train-batch-size 2 --gradient-accumulation-steps 6
modal run --detach twotower_rrf_triplet_ablation/modal_train.py --run-id abl_c_baseline_v2 \
    --rows-filename rrf_003_multineg_k1.json --negatives-per-anchor 1 --train-batch-size 2 --gradient-accumulation-steps 6

modal app logs <app-id>   # fetch mode, never -f
```

## Next steps

1. **Push micro-batch to the ceiling.** It is the lever that works, and 6 was
   not the limit — the probed OOM ceiling was 8 at k=2, and k=1 uses less
   memory, so 8–10 is likely reachable. Cheapest remaining accuracy.
2. **Re-test k=2 without duplicates.** Restrict k=2 rows to the 189 query_keys
   that genuinely have two distinct negatives. Smaller dataset, but it would
   answer whether real extra negatives help, which this ablation could not.
3. **Fix the dev set before trusting selection again.** 60 rows spanning a
   0.50–0.58 metric range cannot reliably rank checkpoints. Either enlarge the
   dev carve or select on a retrieval-shaped metric instead of triplet accuracy.
4. **Replicate the bigbatch corner** so all four cells have error estimates.
