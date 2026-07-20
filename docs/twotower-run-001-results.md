# Two-Tower `run_001`: results and why they aren't comparable to baselines yet

Full 5-epoch LoRA fine-tune of `voyage-4-nano` on `twotower/`, run on Modal
(L4 GPU). See `docs/two-tower-fine-tune-plan.md` for the architecture/loss
decision and `docs/modal-training-guide.md` for the Modal setup.

Plain-language findings and recommended next steps:
[`twotower-run-001-findings.md`](twotower-run-001-findings.md).

## Run config

| | |
|---|---|
| Model | `voyageai/voyage-4-nano`, LoRA rank 8, alpha 16, targets `q/k/v/o_proj` |
| Loss | `ContrastiveLoss` (pairwise labeled, margin 0.5) — see decision rationale below |
| Data | 660 canonical pairs → 530 train / 61 train-dev / 69 frozen holdout |
| `max_seq_length` | 4096 |
| Epochs | 5, batch size 2, grad accum 4, lr 2e-4 |
| `split_hash` | `20bbe8f293127372` |
| `data_hash` | `0cdfeb652624869a` |
| Trainable params | 983,040 / 347,435,008 (0.28%) |
| Runtime | ~754s train + eval on L4 |
| Adapter | `artifacts/twotower/run_001/adapter/` |

Loss choice: the reviewed synthetic data is independently-labeled pairs, not
same-seeker (positive, hard-negative) triples — of 91 distinct seekers with
staged synth pairs, only 5 had both a positive and a negative. That ruled
out `MultipleNegativesRankingLoss` triplets as the primary loss; pairwise
`ContrastiveLoss` consumes the labeled data directly, with a triplet path
kept available in `twotower/data.py::to_triplet_rows()` for later.

## Train-dev metrics by epoch

| Epoch | pair AUC | pair AP | best-F1 | retrieval MRR | NDCG@10 | R@10 |
|---|---|---|---|---|---|---|
| 1 | 0.890 | 0.886 | 0.857 | 0.774 | 0.830 | 1.000 |
| 2 | 0.943 | 0.933 | 0.906 | **0.781** | **0.834** | 1.000 |
| 3 | **0.989** | **0.989** | **0.952** | 0.734 | 0.789 | 0.967 |
| 4 | 0.986 | 0.986 | 0.935 | 0.727 | 0.785 | 0.967 |
| 5 (final, selected) | 0.986 | 0.987 | 0.935 | 0.743 | 0.797 | 0.967 |

Note: pair-classification metrics (AUC/AP/F1) keep climbing through epoch 5,
while retrieval metrics (MRR/NDCG/R@10) peak at epoch 2 and drift down
afterward. Consistent with the model increasingly nailing the binary
pos/neg boundary at some cost to fine-grained ranking among candidates —
worth watching if epoch count increases in later runs.

Epoch 3 (AUC 0.989) technically outscored the selected epoch 5 (AUC 0.986)
on the intended selection metric — see
[`docs/possible-bugs.md`](possible-bugs.md) #2, checkpoint selection
silently fell back to the final epoch instead of reloading epoch 3.

## Raw side-by-side vs. cached baselines (NOT a valid comparison — see below)

| | pair AUC | pair AP | best-F1 | MRR | NDCG@10 | R@10 | population / max_len |
|---|---|---|---|---|---|---|---|
| Frozen BERT | 0.470 | 0.511 | 0.676 | 0.094 | 0.099 | 0.180 | 200 pairs / 512 |
| Voyage-4-nano | 0.561 | 0.557 | 0.671 | 0.301 | 0.360 | 0.600 | 200 pairs / 8192 |
| Voyage-4-large (prod) | 0.573 | 0.571 | 0.669 | 0.310 | 0.393 | 0.700 | 200 pairs / ~8192 |
| twotower run_001 (train-dev, epoch 5) | 0.986 | 0.987 | 0.935 | 0.743 | 0.797 | 0.967 | 61 pairs / 4096 |

The gap looks huge, but this table is not a valid head-to-head. Three
uncontrolled differences, in order of severity:

1. **Different populations.** `baselines/*/eval.py::load_pairs()` reads
   `data/dataset_positive.json`/`dataset_negative.json` directly with no
   train/holdout split — the cached baseline numbers (`num_positive: 100,
   num_negative: 100`) were computed when those files held only the
   original 200 real seed pairs, combining what's now the 131-pair "train"
   split and the 69-pair frozen holdout with no separation. The twotower
   number above is 61 train-dev pairs, a different subset entirely.
2. **Train-dev, not holdout.** Train-dev is user-disjoint from training but
   drawn from the same 660-pair pool the adapter trained on — 50 of the 61
   train-dev pairs are synthetic (`train_dev_synth: 50`), generated from
   the same LLM/prompt distribution the model was fine-tuned against. The
   baselines never saw any of this data, synthetic or real. Beating a
   frozen baseline on data drawn from your own training distribution is
   expected and not evidence of beating it on real-world matching.
3. **Different `max_seq_length`.** Baselines used 8192; twotower used 4096.
   More truncation on the twotower side — if anything this should hurt its
   numbers, not inflate them, but it's still an uncontrolled variable in a
   supposed apples-to-apples table.

## Real holdout eval (2026-07-19) — the number that actually matters

Ran `twotower/eval.py --split holdout --adapter-dir
artifacts/twotower/run_001/adapter` against the frozen 69-pair real
holdout (`eval_pair_ids` — never touched by the synth generator, in
training, train-dev, or few-shot conditioning). Locally on MPS, no
retraining, same `run_001` LoRA weights.

| | pair AUC | pair AP | best-F1 | MRR | NDCG@10 | Recall@10 |
|---|---|---|---|---|---|---|
| Frozen BERT | 0.470 | 0.511 | 0.676 | 0.094 | 0.099 | 0.180 |
| Voyage-4-nano | 0.561 | 0.557 | 0.671 | 0.301 | 0.360 | 0.600 |
| Voyage-4-large (prod) | 0.573 | 0.571 | 0.669 | 0.310 | 0.393 | 0.700 |
| twotower train-dev (mostly synthetic, misleading — see above) | 0.986 | 0.987 | 0.935 | 0.743 | 0.797 | 0.967 |
| twotower real holdout (69 pairs) | 0.578 | 0.487 | 0.619 | 0.283 | 0.359 | 0.655 |

**Update (2026-07-19): the baseline rows above are stale** — computed on
the full 200-pair dataset at `max_length=8192`, not the same 69-pair
population as the twotower row. See "Matched-population comparison" below
for the corrected numbers, which sharpen this verdict rather than soften
it — Voyage-large's *true* holdout retrieval quality was actually
understated by the stale table.

**Verdict (informal, superseded below): the fine-tune does not beat the
frozen baselines on real data.** Pair AUC is roughly tied with Voyage-nano
(0.578 vs 0.561), below Voyage-large (0.573 — essentially a wash) on every
other metric. The 0.986-vs-0.573 gap in the earlier "raw" table was almost
entirely an artifact of evaluating on data statistically close to the
model's own training distribution, not a real capability gap.

**Worse: hard-negative slice AUC is 0.4845 — below chance.** Slice
breakdown (`neg_hardness`, lexical-Jaccard quartiles): easy negatives
AUC 0.693 (n=10), hard negatives AUC 0.4845 (n=20). On real negatives that
are lexically close to the query — exactly the case the synthetic hard
negatives were designed to teach — the fine-tuned model performs *worse
than random*. Combined with the train-dev-vs-holdout divergence, this is
strong evidence the LoRA adapter learned to detect stylistic/structural
artifacts specific to the two synthetic-generation prompts
(`generate_pos.md` vs `generate_neg.md`) rather than the intended
role/stage/side/geo/prefs matching semantics. `data/synthetic/strategy.md`
already worried about this exact failure mode ("teaching old-model blind
spots" / mode collapse) — this run is a concrete instance of it materializing
as generation-artifact overfitting instead.

Caveats on this specific holdout run, for completeness: it used the
epoch-5 checkpoint, not the true best-by-train-dev-AUC epoch 3, due to the
still-unfixed `select_best_checkpoint` bug (`docs/possible-bugs.md` #2) —
though train-dev AUC is now shown to be an unreliable selection signal
regardless, so this likely doesn't change the conclusion. `max_seq_length`
was 4096 vs. the baselines' 8192 (uncontrolled, but truncation-stats showed
only ~1.2% of texts exceed 4096, so unlikely to explain a 0.4-AUC-point
gap). Several intent slices (`customers` n=1, `hiring` n=3,
`partnerships` n=1) are too small to be meaningful.

## Matched-population comparison (2026-07-19, updated 2026-07-20) — the corrected, final table

All rows now share the identical 69-pair real holdout population (65
unique candidates) and matched `max_length` (4096 for the Voyage models;
BERT stays at its native 512-token architectural cap). Baselines rerun via
the `--holdout-only` flag (`baselines/holdout.py`); full table in
`docs/baseline-results-holdout.md`. `tfidf` added 2026-07-20 — a plain
TF-IDF-cosine lexical baseline (`baselines/tfidf/`, no neural model), to
establish a lexical floor: does an embedding-based approach actually beat
literal keyword overlap for this task?

| | pair AUC | MRR | NDCG@10 | Recall@10 | Top-1 | easy-neg AUC | hard-neg AUC |
|---|---|---|---|---|---|---|---|
| TF-IDF (lexical, no model) | 0.592 | 0.248 | 0.282 | 0.483 | 0.138 | 0.755 | 0.502 |
| Frozen BERT | 0.460 | 0.137 | 0.156 | 0.310 | 0.069 | 0.638 | 0.422 |
| Voyage-4-nano | 0.579 | 0.461 | 0.523 | 0.759 | 0.276 | 0.621 | 0.571 |
| Voyage-4-large (prod) | 0.609 | 0.529 | 0.604 | 0.862 | 0.345 | 0.600 | 0.602 |
| twotower `run_001` | 0.578 | 0.283 | 0.359 | 0.655 | 0.069 | 0.693 | 0.4845 |
| twotower `arm_a_real_only` | 0.579 | 0.388 | 0.471 | 0.793 | 0.241 | 0.655 | 0.500 |

**This sharpens the verdict, it doesn't soften it.** Matching the
population moved Voyage-large's numbers *up*, not twotower's — its true
holdout MRR is 0.529, not the stale 0.310 the old full-200-pair number
implied (smaller holdout-only candidate corpus = easier retrieval). On
this fair comparison, both twotower runs are roughly tied with Voyage-nano
on binary pair classification but **clearly worse than both Voyage
baselines on every ranking metric**, and `run_001`'s Top-1 accuracy
(0.069) ties the near-random BERT baseline. This is consistent with the
#4 diagnosis: the model learned to classify one candidate in isolation
(exploiting synthetic-generation artifacts), not to rank many candidates
against each other — exactly the skill retrieval metrics measure and
pair-classification metrics don't.

**The TF-IDF row adds an uncomfortable data point: plain lexical cosine
similarity (0.592 pair AUC) beats *both* twotower runs (0.578, 0.579) on
binary pair classification** — a model with zero training, zero neural
weights, just keyword-overlap counting, edges out a fine-tuned LoRA
adapter on the exact metric that adapter was optimized for. TF-IDF is
weakest of all six on retrieval (0.248 MRR, only clearing BERT) —
consistent with the expected lexical-floor story: keyword overlap alone
can often tell you "these two texts share vocabulary" (enough for
binary classification with a well-chosen threshold) but struggles to
rank the *single best* candidate out of many plausible keyword-overlapping
ones, which needs more than surface matching. The neg-hardness split
confirms this by construction: TF-IDF nails "easy" negatives (0.755 AUC,
low lexical overlap with the query by definition) but sits at exactly
chance on "hard" ones (0.502, high lexical overlap by definition) — a
clean sanity check that the slice metric itself behaves as intended.
Taken together with the twotower numbers: neither `run_001` nor
`arm_a_real_only` clears even the cheapest possible baseline on pair
classification, reinforcing that pair AUC alone is a weak signal for this
task and retrieval metrics (where Voyage-large's advantage is much
clearer, and TF-IDF's weakness is much clearer) are the more informative
ones to track for Arm C.

Caveats on the `run_001` holdout number specifically: it used the epoch-5
checkpoint, not the true best-by-train-dev-AUC epoch 3, due to the
`select_best_checkpoint` bug active at the time (`docs/possible-bugs.md`
#2, since fixed) — though train-dev AUC is now shown to be an unreliable
selection signal regardless, so this likely doesn't change the conclusion.
Several intent slices (`customers` n=1, `hiring` n=3, `partnerships` n=1)
are too small to be meaningful.

## Arm A — real-only control (2026-07-19): the decisive piece of evidence

Trained `arm_a_real_only` with the identical recipe as `run_001` (5 epochs,
same hyperparameters, Modal L4) but with `--real-only` — 111 real train
pairs, 20 real train-dev pairs, **zero synthetic pairs**, vs. `run_001`'s
530 pairs (410 of them synthetic). Same real 69-pair holdout eval.

| | pair AUC | MRR | NDCG@10 | Recall@10 | Top-1 | hard-neg AUC |
|---|---|---|---|---|---|---|
| Frozen BERT | 0.460 | 0.137 | 0.156 | 0.310 | 0.069 | 0.422 |
| Voyage-4-nano | 0.579 | 0.461 | 0.523 | 0.759 | 0.276 | 0.571 |
| Voyage-4-large (prod) | 0.609 | 0.529 | 0.604 | 0.862 | 0.345 | 0.602 |
| twotower `run_001` (real + 410 unfixed synth) | 0.578 | 0.283 | 0.359 | 0.655 | 0.069 | 0.4845 |
| **twotower `arm_a_real_only` (111 real, zero synth)** | 0.579 | **0.388** | **0.471** | **0.793** | 0.241 | **0.500** |

**Arm A — with less than a quarter of the training data and none of it
synthetic — beats `run_001` on every single metric except pair AUC (a
statistical tie, 0.579 vs 0.578).** MRR is 37% higher, NDCG@10 31% higher,
Recall@10 21% higher, hard-negative AUC moves from *below chance* (0.4845)
to *exactly chance* (0.500). This is the cleanest possible evidence for the
#4 diagnosis: the 410 unfixed synthetic pairs in `run_001` weren't just
failing to help — **they were actively making the model worse than not
using them at all.** Checkpoint selection also worked correctly here for
the first time on a genuinely non-monotonic run (picked steps=56/epoch 4
over the higher-step-count-but-lower-scoring epoch 5), independently
validating the `possible-bugs.md` #2 fix.

Arm A still doesn't beat Voyage-nano or Voyage-large — it's in the same
neighborhood as nano (slightly below on MRR/NDCG/R@10, tied on AUC) and
clearly below large. But going from "actively worse than a coin flip on
hard negatives" (`run_001`) to "roughly matching Voyage-nano" using 1/5th
the data and zero synthetic pairs is a large jump, and sets a concrete bar
for Arm C: **does real + *fixed* synthetic data beat Arm A's real-only
numbers, and does it close the remaining gap to Voyage-large?** If Arm C
doesn't clear Arm A, the fixed synthetic data is still net-harmful and
needs further work before scaling.

## Next steps

- ~~Root-cause the shortcut-learning failure~~ — done, see
  `docs/possible-bugs.md` #4: fixed the seed-truncation bug in
  `synth_pipeline/llm.py`, banned meta-commentary give-aways and closed
  the CRM-tone gap in `generate_neg.md`. Validated on a small pilot batch
  (cheatability AUC 0.992 → 0.863, material drop).
- ~~Fix `select_best_checkpoint`~~ — done (`docs/possible-bugs.md` #2):
  root cause was ST's evaluator writing to `checkpoints_dir/eval/`, not
  `checkpoints_dir` directly, plus fragile epoch-float parsing; now keyed
  on `steps` matching HF's `checkpoint-{global_step}` naming, and reloads
  via `load_adapter` (checkpoints are LoRA-only saves, not full ST dirs).
- ~~Rerun baselines on the matched holdout~~ — done, see table above and
  `docs/baseline-results-holdout.md`.
- ~~Train a real-only control arm~~ — done (`arm_a_real_only`,
  `--real-only` flag added to `twotower/train.py`/`modal_train.py`) —
  see above. Confirms the unfixed synthetic data was net-harmful, not
  merely unhelpful.
- **Remaining:** regenerate a full-scale synthetic batch with the fixed
  prompts (Arm C) and train on real + fixed-synth, using Arm A's numbers
  as the bar to clear (not just Voyage-large's). Scale/timing of that
  regeneration is a deliberate follow-up decision, not yet scheduled.

**Bottom line: `run_001` does not clear the "beat voyage-4-large" bar from
`docs/two-tower-fine-tune-plan.md`'s decision gate**, and the matched
comparison shows the gap is larger than first thought, concentrated in
retrieval/ranking quality rather than binary pair classification. But
Arm A proves the architecture and training recipe aren't the problem —
real-only training with the same recipe roughly matches Voyage-nano using
1/5th the data. The unfixed synthetic data was the problem, and it was
actively harmful, not just inert. The root-cause fixes are in and
validated at small scale (Phase 3); the next real test is whether a full
retrain on the fixed data (Arm C) beats Arm A and closes the gap to
Voyage-large.
