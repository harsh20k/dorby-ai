# Do two-field fine-tunes beat the full-profile fine-tune?

## Question

The field-isolation experiment found only three profile fields carry real
person-identity signal alone (`positioning`, `background`, `lookingFor` —
0.85-0.89 own-profile cosine, vs 0.37-0.45 for logistics fields like
`locationAvailability`). A free eval-time sweep (`field_pairs_sweep/`,
frozen voyage-4-nano, seeker side only, candidate kept as the full profile)
then scored all three pairings on the seeker side and found none beat the
existing `concat_baseline`/`query_only` frozen rows.

This experiment asks the next question directly: trim **both** seeker and
candidate to one field pair, actually fine-tune (not just score frozen),
and see whether any of the three pairings beats `top1_ctrl` — the project's
best two-tower fine-tune, trained on the full 8-field profile. Three
separate packages, one per pairing, sharing `top1_ctrl`'s exact recipe so
the only variable is which two fields the text is built from.

## Method

Three isolated packages: `twotower_field_pos_bg/`, `twotower_field_pos_look/`,
`twotower_field_bg_look/` — each a near-copy of `twotower_top1_optimised/`'s
control corner (plain `MultipleNegativesRankingLoss(scale=20.0)`, no
hardness weighting), with the hardness-weighting code removed entirely
(none of the three arms use it). Every setting outside the input text is
held identical to `top1_ctrl_001`:

- **One shared tower**, not separate ones — matching every fine-tune in
  this project. `twotower_split/` already tested giving query and candidate
  independent towers and lost; trimming both sides to the same field pair
  doesn't reopen that question, since the shared tower already handles
  asymmetric content between the two sides via Voyage's own query/document
  role prompts, and every experiment in this project (including this one)
  has asymmetric seeker/candidate text.
- LoRA rank 8 / alpha 16 / dropout 0.05 on q/k/v/o_proj (983,040 trainable
  params), micro-batch 6 / accum 2 (effective batch 12, 245 optimizer
  steps), lr 2e-4, 5 epochs, `voyage-4-nano` at its native 1024-dim
  truncation.
- Same 643-row `rrf_003` population `top1_ctrl` trained on (583 train / 60
  dev after the seeker-disjoint carve, 0% padding) — same query_keys, same
  seekers, same seed. Only the text differs.
- Same checkpoint selection: `CorpusRecallDevEvaluator`, `primary_metric=
  "recall@1"`.

**The one thing that changes per package:** what text goes into
`anchor`/`positive`/`negatives`. Three new row-builder scripts
(`scripts/build_rrf_multineg_triplets_{pos_bg,pos_look,bg_look}.py`, each a
near-copy of `scripts/build_rrf_multineg_triplets.py`) trim **both** seeker
and candidate to one field pair, no query, no other field:

| Package | Fields (both sides) |
|---|---|
| `twotower_field_pos_bg` | positioning + background |
| `twotower_field_pos_look` | positioning + lookingFor |
| `twotower_field_bg_look` | background + lookingFor |

All three row files: 643 rows, 297 seekers, 0% padding — verified identical
population size to `top1_ctrl`'s row file, pinned by
`tests/test_field_{pos_bg,pos_look,bg_look}.py`.

**Scoring problem this required solving:** the real 200-pair loader
(`eval_real_full.eval.run_eval`) always builds candidate text from the full
profile, so it can't score these adapters without feeding them
out-of-distribution input. Each package has its own `eval.py` that reuses
`twotower.eval`'s model loading and query/document encoding unmodified, but
builds seeker *and* candidate text via the same trimmed field-pair builder
the row file used. Hardness split (easy/hard negative) stays pinned to the
full profile+query baseline text in all three, so hard-neg AUC is still
comparable to every other row in the project even though the scored
vectors are trimmed.

```bash
python -m pytest tests/test_field_pos_bg.py tests/test_field_pos_look.py tests/test_field_bg_look.py -q

modal run --detach twotower_field_pos_bg/modal_train.py --run-id field_pos_bg_001
modal run --detach twotower_field_pos_look/modal_train.py --run-id field_pos_look_001
modal run --detach twotower_field_bg_look/modal_train.py --run-id field_bg_look_001

modal run twotower_field_pos_bg/modal_eval.py --run-id field_pos_bg_001
modal run twotower_field_pos_look/modal_eval.py --run-id field_pos_look_001
modal run twotower_field_bg_look/modal_eval.py --run-id field_bg_look_001
```

## Results — all 200 real pairs

| Approach | Pair AUC | Hard-neg AUC | Easy-neg AUC | MRR | R@1 | R@5 | R@10 |
|---|---|---|---|---|---|---|---|
| `top1_ctrl` (best full-profile fine-tune) | **0.5683** | 0.5484 | **0.6836** | **0.3550** | **0.19** | — | **0.69** |
| `field_bg_look_001` (background+lookingFor) | 0.5610 | 0.5786 | 0.6024 | 0.2674 | 0.12 | 0.43 | 0.58 |
| `field_pos_bg_001` (positioning+background) | 0.5496 | 0.5720 | 0.5940 | 0.2497 | 0.13 | 0.36 | 0.48 |
| `field_pos_look_001` (positioning+lookingFor) | 0.5467 | **0.5794** | 0.5932 | 0.2407 | 0.13 | 0.36 | 0.50 |

Reference rows (frozen voyage-4-nano, seeker side only trimmed, candidate
kept as the **full** profile — the free eval-time sweep that motivated this
experiment):

| Frozen sweep arm | Pair AUC | Hard-neg AUC | MRR | R@1 | R@10 |
|---|---|---|---|---|---|
| pos_lookingfor (seeker only) | 0.5386 | 0.4916 | 0.2879 | 0.15 | 0.56 |
| background_lookingfor (seeker only) | 0.5367 | 0.5164 | 0.2426 | 0.12 | 0.51 |
| pos_background (seeker only) | 0.5412 | 0.5278 | 0.2060 | 0.10 | 0.40 |

## Findings

**None of the three beats `top1_ctrl`.** All three lose on pair AUC, MRR,
R@1, and R@10 — the metrics that matter most for a search product.
`field_bg_look_001` comes closest (AUC 0.561 vs 0.568, only 0.007 behind)
but still trails badly on MRR (0.267 vs 0.355) and R@10 (0.58 vs 0.69).

**The ranking flipped once training and a trimmed candidate side entered
the picture.** The frozen sweep (seeker-only trimmed, full-profile
candidate) ranked `pos_lookingfor` clearly best of the three. Once both
sides are trimmed and the model is actually fine-tuned,
`background_lookingfor` wins instead — on every metric. This means the
frozen seeker-only sweep was not a reliable predictor of which field pair
would fine-tune best once the candidate side changed too; the two questions
("which fields help a frozen encoder read the seeker" vs. "which fields
give a trainable adapter the most to work with on both sides") have
different answers here.

**Training helped, consistently.** All three trained arms beat their own
frozen seeker-only-trimmed counterpart on every retrieval metric (e.g.
`field_pos_bg_001`'s MRR 0.2497 vs the frozen sweep's 0.2060) — trimming to
two fields isn't a dead end, fine-tuning recovers real signal, just not
enough to close the gap to the full-profile fine-tune.

**All three beat `top1_ctrl` on hard-negative AUC** — the one consistent
win across every arm (0.572-0.579 vs `top1_ctrl`'s 0.548), traded against a
large drop in easy-neg AUC (0.593-0.602 vs 0.684). Less text means less
surface area to get fooled by superficial topic overlap on plausible-but-
wrong candidates, but also less to work with for general ranking. This
mirrors the pattern already seen in the single-field-pair pilot
(`field_pos_bg_001` alone) and now holds across all three pairings — it
looks like a property of trimming to two identity fields in general, not
a quirk of one particular pair.

**Holdout misled again, on all three.** The 69-pair holdout ranked
`field_bg_look_001` best by a wide margin (AUC 0.6224, R@1 0.31) — a number
that would have looked like a clean win over `top1_ctrl`'s own holdout
result if this experiment had stopped there. Scored on all 200, the gap to
`top1_ctrl` reopens on every metric except hard-neg AUC. Same standing
project rule, held again: score on all 200 before calling anything a
result.

## What this means

Two-field seeker+candidate text is a genuine trade, not a strict loss:
worse general ranking, better hard-negative resistance, and none of the
three pairings closes the gap to the full-profile fine-tune on the metrics
that matter most for retrieval. `background_lookingfor` is the strongest of
the three tested — worth remembering if a future experiment wants to
combine field trimming with something else (e.g. the query, which none of
these three arms used at all), but on its own it is not a replacement for
`top1_ctrl`'s full-profile recipe.

`top1_ctrl` plus eval-time query-weighting
(`docs/twotower-no-query-experiment.md`, recall@1 0.32 with zero extra
training) remains the strongest, cheapest lever found in this project —
reinforced again by this result, since none of these three genuinely new
fine-tunes caught up to it either.
