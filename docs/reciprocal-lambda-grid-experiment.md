# Lambda sensitivity sweep (no fitting) — does a diagnostic curve back up the earlier claims?

**Status: run complete (`voyage-4-nano`, Modal A10G, run `real200_001`).**
Chart: [`docs/html/reciprocal-lambda-grid.html`](html/reciprocal-lambda-grid.html)
([published](https://claude.ai/code/artifact/c522e5a2-a1fb-4a77-8f96-ed8e9c403431)).
No
fitting step this time — `lambda` is swept from -2 to +2 (step 0.05) and pair
ROC-AUC of `s_fwd + lambda*s_recip` is reported directly against the real 200
(and, separately, the 69-pair holdout) at every grid point. The curve is
smooth and unimodal, peaks on the **positive** side of zero for both
populations, and both peaks beat forward-only — directionally consistent
with the original `reciprocal_static` finding (λ fit on real train, λ=1.75)
and the opposite of the `reciprocal_static_rrf003` null result (λ fit on
rrf_003 judge labels, λ=0.05). **Caveat that matters more than the headline
here: every number below, including "best lambda," comes from looking
directly at the same labels being scored — this is a sensitivity curve, not
a validated estimate**, unlike the other two experiments' train-only or
rrf003-only fits.

## What this is, and isn't

Both prior reciprocal experiments picked a single `lambda` by *fitting* it —
grid-search maximizing AUC on one population, then freezing it and checking a
different, held-out population:

- `docs/reciprocal-static-experiment.md`: fit on 131 real train pairs → λ=1.75 → real-200 AUC 0.5964 (full `bg_text` field set).
- `docs/reciprocal-static-rrf003-experiment.md`: fit on 2,619 rrf_003 judge-labeled pairs → λ=0.05 → real-200 AUC 0.5573 (narrowed `bg_text`, positioning+background only).

This experiment (`baselines/reciprocal_lambda_grid/`, new isolated package)
does neither. It just evaluates the combined score at every λ in the grid,
directly against the population being reported — no train/holdout split, no
frozen value, no claim of a deployable number. `bg_text` stays narrowed to
`positioning`+`background`, matching `reciprocal_static_rrf003`'s field
choice (not the original's wider set).

## Results

| population | forward-only AUC (λ=0) | curve max | Δ at curve max |
|---|---|---|---|
| holdout (69 pairs) | 0.5966 | λ=0.25 → 0.6181 | +0.0215 |
| all 200 | 0.5578 | λ=0.55 → 0.5663 | +0.0085 |

Curve shape (quarter-lambda steps, full 0.05-step grid in `artifacts/reciprocal_lambda_grid/metrics.json`):

| λ | holdout AUC | all-200 AUC |
|---|---|---|
| -2.00 | 0.4871 | 0.4933 |
| -1.00 | 0.5224 | 0.5091 |
| -0.50 | 0.5759 | 0.5352 |
| **0.00 (forward-only)** | **0.5966** | **0.5578** |
| +0.25 | **0.6181** (peak) | 0.5618 |
| +0.50 | 0.6052 | **0.5663** (peak) |
| +1.00 | 0.6086 | 0.5582 |
| +2.00 | 0.5871 | 0.5493 |

Both curves are smooth, single-peaked, and asymmetric: negative λ degrades
AUC sharply and monotonically (down to ~0.49, chance, at λ=-2), while
positive λ gives a mild, real lift before decaying slowly past λ≈1. That
shape is itself informative — a reciprocal-score direction that carried no
signal at all would produce a flat or noisy curve, not this one.

## Reading the three reciprocal experiments together

- **All three curves agree on the sign.** `s_reciprocal` added with a
  positive coefficient helps; subtracted, it hurts a lot. This is the one
  thing that hasn't flipped across a train-fit, a synthetic-fit, and a
  no-fit diagnostic sweep.
- **They disagree hard on magnitude, and on the same axis that's already
  confounded.** λ=1.75 (real-train fit, full `bg_text`) landed near this
  sweep's *decaying* region, not its peak (λ≈0.25–0.55, narrowed
  `bg_text`) — consistent with `reciprocal-static-rrf003-experiment.md`'s
  finding that narrowing the field set alone lowers the achievable AUC,
  which could also be shifting where the optimal λ sits. This sweep, run only
  with narrowed fields, can't separate "the optimal λ moved because the
  fitting population changed" from "the optimal λ moved because the field set
  changed" — same limitation flagged in the rrf003 doc, still unresolved.
- **The rrf003 fit (λ=0.05) undershot this sweep's own peak on the same field
  set** (0.05 vs. the ≈0.25–0.55 region this curve shows is actually best) —
  further evidence that experiment's null result was a bad fit, not evidence
  the reciprocal term itself is worthless.
- **None of this sweep's numbers are validated.** The "best" λ here is read
  off the same labels being scored, which is optimistic by construction (like
  reporting train accuracy, not test accuracy). The original
  `reciprocal_static` run remains the only one of the three with a proper
  train→holdout separation, and its own bootstrap CI already crossed zero —
  so "the curve looks nice" here should raise, not settle, confidence.

## What would actually resolve this

Re-run `reciprocal_static`'s exact recipe (train-only fit, held-out check)
with `bg_text` narrowed to `positioning`+`background`, so field set and
fitting discipline are no longer both moving at once. That isolates whether
the field narrowing itself is what's shrinking the gain, independent of
which population picks λ.

## Repro

```bash
modal run baselines/reciprocal_lambda_grid/modal_eval.py --run-id real200_001
modal volume get dorby-reciprocal-lambda-grid-eval real200_001/metrics.json \
    ./artifacts/reciprocal_lambda_grid/metrics.json

# local (small/cheap enough for MPS)
python -m baselines.reciprocal_lambda_grid.eval --data-dir data

pytest tests/test_reciprocal_lambda_grid.py -q
```
