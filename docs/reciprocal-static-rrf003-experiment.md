# Reciprocal static score, lambda calibrated on rrf_003 judge labels

**Status: run complete (`voyage-4-nano`, Modal A10G, run `real200_001`).**
Calibrating `lambda` on 2,619 judge-labeled synthetic pairs instead of the 131
real train pairs, and narrowing the background view to `positioning` +
`background` only, **does not reproduce the gain** the original
[`docs/reciprocal-static-experiment.md`](reciprocal-static-experiment.md) run
found. The combined score is statistically indistinguishable from
forward-only on the real 200, and both are well below the original run's
numbers on the same population.

## What changed vs. the original experiment

New isolated package **`baselines/reciprocal_static_rrf003/`** (per this
repo's experiment-isolation rule — `baselines/reciprocal_static/` was not
touched):

1. **Lambda fitting population**: 1-D grid search maximizing pair ROC-AUC of
   `s_fwd + lambda*s_recip`, same procedure as the original, but on
   `exports/rrf_datasets/rrf_003`'s 2,619 pairs (1,175 pos / 1,444 neg),
   labeled by an LLM judge (`google/gemini-3.1-flash-lite`, naive framing —
   0.6358 pair AUC / 0.5942 decision accuracy as a labeler on the real
   holdout), not by real train-set accept/decline outcomes.
2. **Background text (`bg_text`)** narrowed to `positioning` + `background`
   only (`baselines/reciprocal_static_rrf003/text.py`), instead of every
   non-`lookingFor` profile field (`notes`, `locationAvailability`,
   `introPreferences`, `personalPreferences`,
   `meetingAndSchedulingPreferences` are now excluded from `v_i`).
3. **`look_text`/`seeker_look_text` unchanged** from the original — pinned
   identical by `tests/test_reciprocal_static_rrf003.py`.

The frozen `lambda` is then applied, unchanged, to the full 200 real
accept/decline pairs — real labels never touch fitting, same discipline as
every other experiment in this repo.

## Results

Fitted on 2,619 rrf_003 pairs: **`lambda = 0.05`** (rrf_003 AUC forward-only
0.5914 → combined 0.5917 — the grid search barely moved off zero).

| population | pair AUC fwd-only | pair AUC combined | Δ | MRR (fwd-only retrieval) | R@1 |
|---|---|---|---|---|---|
| rrf_003 (fitting population) | 0.5914 | 0.5917 | +0.0003 | — | — |
| real 200 (frozen lambda) | 0.5578 | 0.5573 | **-0.0005** | 0.3063 | 0.1300 |

Neg-hardness slice on real 200 (combined score): easy-neg AUC 0.682, hard-neg
AUC 0.5236 — the usual pattern (easy negatives are far easier than hard ones),
nothing like the original run's inversion.

## Why this run is a null result, not a replication

**The reciprocal term barely exists in this run.** `lambda=0.05` is close to
zero — the grid search on rrf_003 found almost no benefit to adding
`s_reciprocal` at all, so `combined ≈ forward-only` by construction. Compare
the original real-train-fit run's `lambda=1.75`, a genuinely large
coefficient that meaningfully reweighted the score.

**Both scores are also worse in absolute terms than the original run** on
the *same* real-200 population: forward-only AUC here is 0.5578 vs. the
original's 0.5638, and combined is 0.5573 vs. 0.5964. Since forward-only
should be identical between the two experiments' shared machinery — same
`look_text`, same encoder, same population — the 0.5578-vs-0.5638 gap is
almost certainly the narrowed `bg_text` (`positioning`+`background` only):
fewer fields means less text for the encoder to work with, which by itself
made the forward retrieval score slightly worse here. **This run therefore
confounds two changes at once (fitting population *and* field set) and can't
cleanly isolate which one caused the drop** — a real limitation of this
experiment, not just of the result.

**Read together with the original run's own caveat** (bootstrap 95% CI on
the AUC delta crossed zero, `P(delta>0)` 0.76–0.84) — this is consistent with
that finding being noise in the first place: a differently-fit lambda on a
20x larger population lands at a coefficient close to zero and a delta with
the opposite sign.

## What this does and doesn't tell you

- **Judge-labeled synthetic pairs did not successfully substitute for real
  labels in this calibration.** The lambda fit on rrf_003 (0.05) and the
  lambda fit on real train (1.75, original experiment) are wildly different
  values, and the rrf_003-fit one does not transfer to a real-AUC gain.
- **This is not a clean test of "does narrowing bg_text help or hurt"** — the
  fitting-population change is entangled with it. Re-run with the original
  full `bg_text` field set and the rrf_003-fit lambda to isolate that half.
- **Not a clean test of "is rrf_003 lambda actually predictive here"
  either** — same entanglement, other direction. Re-run with narrowed
  `bg_text` and the *real-train*-fit lambda (i.e., just the original
  experiment's exact recipe on this narrower field set) to isolate that half.

## Repro

```bash
modal run baselines/reciprocal_static_rrf003/modal_eval.py --run-id real200_001
modal volume get dorby-reciprocal-static-rrf003-eval real200_001/metrics.json \
    ./artifacts/reciprocal_static_rrf003/metrics.json

# local (small/cheap enough for MPS)
python -m baselines.reciprocal_static_rrf003.eval --data-dir data

pytest tests/test_reciprocal_static_rrf003.py -q
```
