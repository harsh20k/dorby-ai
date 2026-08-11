# Lambda sensitivity sweep on the newest fine-tuned model (voyage_gemini_ctrl)

**Status: run complete (`voyage_gemini_ctrl_001` LoRA adapter on
voyage-4-nano, Modal A10G, run `real200_001`).** Same no-fitting λ sweep as
`docs/reciprocal-lambda-grid-experiment.md` and
`docs/reciprocal-lambda-grid-top1ctrl-experiment.md`, now on
`voyage_gemini_ctrl_001` — `top1_ctrl`'s exact training recipe, retrained on
a larger, newer, but measurably leakier synthetic batch
(`docs/twotower-voyage-gemini-ctrl-experiment.md`). **Forward-only alone hits
0.7164 holdout AUC — the best number of any kind (fitted, fine-tuned, or
swept) recorded anywhere in this project so far — and unlike `top1_ctrl`,
the reciprocal term keeps the same positive sign as frozen voyage-4-nano on
both populations, no sign flip.**

## Same out-of-distribution caveat as top1_ctrl

`voyage_gemini_ctrl` was trained with the identical text builders as
`top1_ctrl` — full-profile seeker/candidate text
(`baselines.bert_frozen.text`), never this experiment's narrower `lookingFor`
+ query / `positioning`+`background` split. Every number below measures how
that fine-tune behaves scored outside its training distribution, same
caveat as the `top1_ctrl` sweep.

**Additional caveat specific to this checkpoint's training data**: the
batch it was trained on (`pairing_voyage_gemini`'s `smoke_test_002`) had its
own leakage checks flag it as measurably leakier than `rrf_003` (the batch
`top1_ctrl` trained on) — candidate-profile-only AUC 0.758 vs. 0.634,
seeker-identity-only AUC 0.780 vs. 0.687, 43.5% of seekers all-positive or
all-negative vs. 30% — see `docs/twotower-voyage-gemini-ctrl-experiment.md`.
That doesn't necessarily explain a gain on the *real* 200-pair population
(leakage in synthetic training data doesn't automatically transfer to real
labels), but it's a live possibility worth flagging before treating this
number as a settled win.

## What changed vs. the other two sweeps

New isolated package **`baselines/reciprocal_lambda_grid_voyage_gemini_ctrl/`**
(neither `baselines/reciprocal_lambda_grid/` nor
`baselines/reciprocal_lambda_grid_top1ctrl/` touched). `text.py` is
byte-identical to both prior sweeps' (`positioning`+`background` only for
`bg_text`, pinned by test). `encode.py::VoyageGeminiCtrlEncoder` is the same
shape as `Top1CtrlEncoder`, pointed at
`artifacts/twotower_voyage_gemini_ctrl/voyage_gemini_ctrl_001/adapter`
instead. No fitting: λ swept from −2 to +2 (step 0.05, 81 points), pair
ROC-AUC of `s_fwd + λ·s_rev` reported directly against the labels being
scored.

## Results

| population | forward-only AUC (λ=0) | curve max | Δ at curve max |
|---|---|---|---|
| holdout (69 pairs) | **0.7164** | λ=0.35 → **0.7345** | +0.0181 |
| all 200 | 0.6108 | λ=1.90 → 0.6587 | +0.0479 |

Curve shape (quarter-λ steps, full 0.05-step grid in
`artifacts/reciprocal_lambda_grid_voyage_gemini_ctrl/metrics.json`):

| λ | holdout AUC | all-200 AUC |
|---|---|---|
| -2.00 | 0.4155 | 0.4082 |
| -1.00 | 0.5224 | 0.4666 |
| -0.50 | 0.6284 | 0.5362 |
| **0.00 (forward-only)** | **0.7164** | **0.6108** |
| +0.25 | 0.7259 | 0.6347 |
| +0.50 | **0.7302** | 0.6478 |
| +1.00 | 0.7164 | 0.6556 |
| +1.75 | 0.7078 | 0.6583 |
| +2.00 | 0.7043 | **0.6576** |

## Reading it against the other two sweeps

- **Both curves keep the same positive sign as frozen voyage-4-nano.**
  Unlike `top1_ctrl`, whose holdout curve flipped to a negative-λ peak, both
  `voyage_gemini_ctrl` populations peak on the positive side — the reciprocal
  term helps in the direction every experiment except `top1_ctrl`'s holdout
  agrees on.
- **The holdout curve is genuinely peaked** (rises to λ≈0.35–0.5, then falls
  back toward forward-only by λ=1 and keeps falling). **The all-200 curve is
  not** — it rises steadily and then plateaus flat from λ≈1.0 onward
  (0.6556 → 0.6587, a 0.003 spread across the last 20 grid points). The
  reported "best λ=1.90" is not a real peak; it's noise on top of a flat
  shelf, and should not be read as meaningfully different from λ=1.0 or
  λ=1.5.
- **Forward-only alone (λ=0) is the headline, not the reciprocal term.**
  0.7164 holdout / 0.6108 all-200 already beat every number in this project
  to date on their own — the reciprocal sweep adds a real but comparatively
  small lift on top (+0.018 holdout, and an unreliable +0.048 all-200 given
  the plateau above).

## What would actually resolve this

Same open items as the `top1_ctrl` sweep: fine-tune directly on the look/bg
split to remove the out-of-distribution confound, and separately, verify
this checkpoint's real-200 gain isn't an artifact of its leakier training
batch by comparing against a `voyage_gemini_ctrl` variant trained on a
cleaned/reviewed batch once one exists.

## Repro

```bash
modal run baselines/reciprocal_lambda_grid_voyage_gemini_ctrl/modal_eval.py --run-id real200_001
modal volume get dorby-reciprocal-lambda-grid-voyage-gemini-ctrl-eval real200_001/metrics.json \
    ./artifacts/reciprocal_lambda_grid_voyage_gemini_ctrl/metrics.json

pytest tests/test_reciprocal_lambda_grid_voyage_gemini_ctrl.py -q
```
