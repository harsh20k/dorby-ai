# Targeting recall@1 directly — one change failed, one won quietly

No fine-tune in this project has improved top-1 retrieval on real data. On all
200 real pairs (`docs/eval-real-full-experiment.md`):

    frozen voyage-4-nano   recall@1 = 0.1800   (18 of 100 queries)
    Arm A, fine-tuned      recall@1 = 0.1800   (+0.0000)

The best recall@1 anywhere in this project belongs to an **untrained** model.
Arm A did improve the rest of the ranking substantially — mean rank 17.9 → 12.1,
median 7 → 5, recall@10 59 → 64 — so it pulls the right person up the list
without converting that into first place.

This experiment tested two changes aimed squarely at that conversion, accepting
in advance that pair AUC might suffer. **One backfired badly. The other produced
the best nano model measured so far — but only shows it on the full 200 pairs;
on the 69-pair holdout it looks like nothing happened.**

Isolated package `twotower_top1_optimised/`. Nothing under `twotower/`,
`twotower_rrf_triplet*/`, or `twotower_qwen_bigbatch/` was modified.

## What was changed, and why those two things

**Change 1 — sharpen the loss toward the top competitor.** Every prior run in
this project called `MultipleNegativesRankingLoss(model=model)` with pure
defaults: `scale=20.0`, `hardness_mode=None`, `hardness_strength=0.0`. The
softmax temperature had never been tuned and the library's built-in
hard-negative weighting had never been switched on. `scale` is an inverse
temperature; raising it makes the distribution peakier so gradient concentrates
on the *highest-scoring* negative rather than spreading across all of them —
mechanically, "beat whoever is currently ranked first". Set to `scale=50.0`,
`hardness_mode="hard_negatives"`, `hardness_strength=1.0`.

**Change 2 — let checkpoint selection see recall@1.** The ablation's dev
evaluator reported `beat_all_accuracy`: did the positive out-score *its own* one
or two negatives. That is a triplet test over 2–3 candidates, while the real
metric ranks against ~178. `CorpusRecallDevEvaluator` now ranks each dev anchor
against the full dev corpus (86 unique candidates over 60 rows) and delegates
scoring to `baselines.metrics.retrieval_metrics` — the same function the real
holdout uses — so dev and holdout compute recall@1 identically by construction.
`primary_metric` is `recall@1`.

Everything else is held identical to Arm A (`abl_a_batch_only_v2`): same
`rrf_003_multineg_k1.json` rows, micro-batch 6, accum 2 (effective batch 12 →
245 optimizer steps), lr 2e-4, 5 epochs, `save_total_limit=5`.

Three runs form a ladder, so each change is attributable:

| run | loss | selection |
|---|---|---|
| `abl_a_batch_only_v2` *(existing)* | default (scale 20) | triplet accuracy |
| `top1_ctrl_001` | default (scale 20) | **recall@1** |
| `top1_001` | **scale 50 + hardness** | **recall@1** |

## Results — real 69-pair holdout

| run | pair AUC | hard-neg | MRR | R@1 | R@10 |
|---|---|---|---|---|---|
| Arm A v2 (baseline) | 0.5983 | 0.6034 | 0.5326 | **0.3793** | 0.8621 |
| `top1_ctrl` — selection fix only | 0.5974 | **0.6121** | **0.5436** | **0.3793** | 0.8621 |
| `top1_sharp` — + sharpened loss | 0.5629 | 0.5241 | 0.4735 | 0.2759 | 0.7241 |

## Results — all 200 real pairs (corpus 178, 100 positive queries)

Ranked by MRR, against every other model measured on this population:

| model | pair AUC | hard-neg | MRR | R@1 | R@10 |
|---|---|---|---|---|---|
| **`top1_ctrl`** | 0.5683 | 0.5484 | **0.3550** | **0.1900** | 0.6900 |
| Arm A v2 | 0.5594 | **0.5558** | 0.3341 | 0.1800 | 0.6400 |
| nano frozen | 0.5593 | 0.5046 | 0.3171 | 0.1800 | 0.5900 |
| voyage-4-large | 0.5726 | 0.5422 | 0.3102 | 0.1300 | **0.7000** |
| qwen micro-6 | **0.5947** | 0.5608 | 0.3031 | 0.1400 | 0.6600 |
| `top1_sharp` | 0.5429 | 0.4578 | 0.3010 | 0.1400 | 0.6400 |
| qwen micro-1 | 0.5604 | 0.4828 | 0.2734 | 0.1400 | 0.5600 |
| qwen frozen 4096 | 0.5420 | 0.4572 | 0.2031 | 0.0400 | 0.5800 |

Deltas against Arm A v2, the previous best nano configuration, on each population:

| | Δ pair AUC | Δ hard-neg | Δ MRR | Δ R@1 | Δ R@10 |
|---|---|---|---|---|---|
| `top1_ctrl`, all 200 | +0.0089 | −0.0074 | **+0.0208** | +0.0100 | **+0.0500** |
| `top1_ctrl`, train 131 | +0.0082 | −0.0362 | **+0.0281** | +0.0141 | +0.0282 |
| `top1_ctrl`, holdout 69 | −0.0009 | +0.0087 | **+0.0110** | +0.0000 | +0.0000 |
| `top1_sharp`, all 200 | −0.0165 | −0.0980 | −0.0332 | −0.0400 | +0.0000 |

## Verdict

**1. Sharpening the loss failed, and failed on its own terms.** The trade offered
was "lose pair AUC, gain recall@1". What actually happened was a loss on *every*
metric — on the holdout, recall@1 0.3793 → 0.2759 (three fewer queries of 29),
MRR −0.070, hard-negative AUC −0.088, recall@10 −0.138, pair AUC −0.035; and on
all 200, recall@1 0.1800 → 0.1400, below even the untrained frozen model, with
hard-negative AUC collapsing to 0.4578 — worse than chance.

The likely mechanism: at micro-batch 6 there are only ~11 in-batch competitors.
A peakier softmax concentrates nearly all gradient on a single negative per step,
and `hardness_strength=1.0` compounds that by up-weighting the explicit hard
negative on top. That is a very high-variance signal — and the negatives it
fixates on carry judge labels that are **59.4% accurate on the hard slice**. The
sharpening worked as designed; it sharpened toward label noise.

This is a real constraint on the whole "make the loss more top-1 focused" family
of ideas: **with noisy labels and few in-batch competitors, concentrating
gradient makes things worse, not better.** Any future attempt in this direction
should first increase the number of competitors (GradCache / cross-batch
negatives) or clean the labels — not sharpen what is already there.

**2. Recall@1-aware selection produced the best nano model measured — and the
holdout hid it.** On the 69-pair holdout it looked outcome-neutral: identical
recall@1, MRR +0.011. On all 200 real pairs it is the **best MRR of any model in
this project (0.3550)**, ahead of Voyage-4-large's 0.3102 and Qwen micro-6's
0.3031, with recall@10 +0.0500 over Arm A and pair AUC +0.0089.

It is also the **first fine-tuned model to beat frozen nano on recall@1 on the
full real set** — 0.1900 vs 0.1800. That is exactly one query out of 100, which
is precisely the resolution limit, so it must not be called significant. But the
direction is no longer zero, and the MRR gain is consistent in sign across all
three populations (+0.021 / +0.028 / +0.011).

**I initially wrote this arm off as neutral based on the holdout alone.** That
was wrong, and it is the third time in this project the 69-pair holdout has
misled a conclusion. The rule is now unambiguous: no arm gets a verdict until it
has been scored on all 200.

A second reason the change matters: **the new dev metric ranked the two arms
correctly for the first time.** Dev recall@1 was 0.3333 (`ctrl`) vs 0.3000 (`sharp`), and the
holdout followed with 0.3793 vs 0.2759. The old `beat_all_accuracy` had been
*anti*-correlated with holdout performance in both prior experiments — it rated
Arm C above Arm A in the nano ablation, and Qwen micro-1 above micro-6, and was
wrong both times. This project now has a directionally trustworthy dev signal,
which is a prerequisite for any future tuning — and here it also picked a
genuinely better model.

**3. Dev recall@10 is already useless.** It saturates at 1.0 on `ctrl` — the
86-candidate dev corpus is too easy at depth 10. Only dev recall@1 and MRR carry
information at this corpus size. If the dev signal is to be leaned on harder, the
dev corpus needs to grow.

## Training loss

| run | loss first → last |
|---|---|
| `top1_ctrl_001` | 1.0163 → 0.6473 |
| `top1_001` | 1.2918 → 0.7617 |

Not comparable across the two: `scale` multiplies the logits, so the sharpened
arm's loss sits on a different scale by construction. Only the within-arm trend
is meaningful, and both trained cleanly.

## Caveats

- One run per arm; no replicate, so no error estimate for this experiment. The
  ablation's arms B and C give the closest reference (pair AUC ±0.0017 and
  ±0.0164); recall@1 on the holdout moves in steps of 1/29 = 0.0345, so
  `sharp`'s −0.1034 is three whole queries and comfortably outside noise, while
  `ctrl`'s +0.0000 is exact.
- Only one point in the sharpening space was tested (`scale=50`,
  `hardness_strength=1.0`). A gentler setting might behave differently; this run
  does not rule that out, it rules out the aggressive corner.
- Training labels remain an LLM judge's opinion on synthetic profiles. Scoring is
  on real pairs only.
- The dev corpus is 86 candidates against the real eval's 178 — the right shape,
  but not the same difficulty.

## Reproduce

```bash
python -m pytest tests/test_top1_optimised.py -q

modal run --detach twotower_top1_optimised/modal_train.py --run-id top1_001
modal run --detach twotower_top1_optimised/modal_train.py --run-id top1_ctrl_001 \
    --loss-scale 20.0 --hardness-mode ""

modal volume get dorby-twotower-top1-checkpoints <run_id> \
    ./artifacts/twotower_top1_optimised/<run_id>
modal run eval_real_full/modal_eval.py --run-id real200_top1 \
    --configs top1_ctrl,top1_sharp
```

Keep the launching shell alive (`wait`) — `--detach` is still cancelled when the
client process is terminated.

## What this leaves

Of the five candidate changes for moving recall@1, this experiment tested the two
cheapest and retired one of them. Remaining, in order of expected value:

1. **Mine the negatives that actually beat us** (ANCE-style iterative
   hard-negative mining). Untouched, and the result here strengthens the case:
   the problem is not that the loss ignores hard negatives, it is that the hard
   negatives it has are the wrong ones — retrieved once, by a different model,
   from a synthetic pool.
2. **More competitors per step** (GradCache / cross-batch memory). Micro-batch is
   the only lever proven to work twice in this project, and this experiment
   suggests why sharpening cannot substitute for it: concentration without volume
   amplifies noise.
3. **Cleaner labels at the decision boundary.** 59.4% judge accuracy on hard
   pairs is a hard ceiling on anything that tries to learn fine distinctions at
   the top of the ranking.
