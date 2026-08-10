# Lambda sensitivity sweep on the fine-tuned model (top1_ctrl)

**Status: run complete (`top1_ctrl_001` LoRA adapter on voyage-4-nano, Modal
A10G, run `real200_001`).** Same no-fitting λ sweep as
`docs/reciprocal-lambda-grid-experiment.md`, with one substitution: every
embedding comes from `top1_ctrl_001` — this project's best fine-tuned
two-tower checkpoint (`docs/twotower-rrf-triplet-ablation-experiment.md`) —
instead of frozen `voyage-4-nano`. **Two findings: the fine-tune's forward-only
score alone is far stronger than frozen (holdout 0.6655 vs. 0.5966, all-200
0.5890 vs. 0.5578), and the reciprocal term's sign flips on holdout — the
curve there peaks at *negative* λ (−0.25), not positive** — the opposite
direction from every other reciprocal experiment in this project so far.

## Caveat before anything else: this text is out-of-distribution for the fine-tune

`top1_ctrl_001` was trained on **full-profile** seeker/candidate text
(`baselines.bert_frozen.text.seeker_to_text`/`candidate_to_text` — every
profile field, tagged, plus the search query on the seeker side; see
`twotower/data.py`). This experiment feeds it the reciprocal split instead:
`k` = `lookingFor` (+ query for the seeker), `v` = `positioning`+`background`
only. **The LoRA adapter has never seen this narrower text shape during
training.** Any number here is a measurement of "how does a model fine-tuned
on one text distribution behave when scored on a different one," not a clean
test of whether fine-tuning helps the reciprocal mechanism specifically. A
fair comparison would fine-tune (or at least evaluate) on the same look/bg
split throughout — untested here.

## What changed vs. the frozen-model sweep

New isolated package **`baselines/reciprocal_lambda_grid_top1ctrl/`**
(`baselines/reciprocal_lambda_grid/` untouched, per the isolation rule).
`text.py` is byte-identical to the frozen sweep's (`positioning`+`background`
only for `bg_text`, pinned by test). The only real change is
`encode.py::Top1CtrlEncoder` — loads `voyage-4-nano` + the
`top1_ctrl_001` LoRA adapter once via `twotower.eval.load_model_for_eval`
(unmodified, already on `main`) and disk-caches encodes the same way
`VoyageNanoEncoder` does, making it a drop-in replacement in the sweep's
scoring code. No fitting: λ swept from −2 to +2 (step 0.05, 81 points),
pair ROC-AUC of `s_fwd + λ·s_rev` reported directly against the labels being
scored, same as the frozen sweep.

## Results

| population | forward-only AUC (λ=0) | curve max | Δ at curve max |
|---|---|---|---|
| holdout (69 pairs) | 0.6655 | **λ=−0.25 → 0.6853** | +0.0198 |
| all 200 | 0.5890 | λ=0.50 → 0.5961 | +0.0071 |

Curve shape (quarter-λ steps, full 0.05-step grid in
`artifacts/reciprocal_lambda_grid_top1ctrl/metrics.json`):

| λ | holdout AUC | all-200 AUC |
|---|---|---|
| -2.00 | 0.5302 | 0.4723 |
| -1.00 | 0.5948 | 0.5102 |
| -0.50 | 0.6690 | 0.5612 |
| **0.00 (forward-only)** | **0.6655** | **0.5890** |
| +0.25 | 0.6586 | 0.5951 |
| +0.50 | 0.6397 | **0.5961** (peak) |
| +1.00 | 0.6095 | 0.5920 |
| +2.00 | 0.5741 | 0.5839 |

## Reading it against the frozen-model sweep

- **Forward-only alone already beats every combined score the frozen sweep
  found.** `top1_ctrl` forward-only holdout AUC (0.6655) beats even the
  frozen sweep's own combined-score peak (0.6181) — the fine-tune is doing
  real work on its own, independent of the reciprocal mechanism.
- **The holdout curve's sign flipped.** Frozen: holdout peaked at λ=+0.25.
  Fine-tuned: holdout peaks at λ=−0.25 — a mirror image around zero, and the
  curve is *asymmetric*: it falls off faster on the positive side (0.6655→
  0.5741 by λ=+2) than the negative side (0.6655→0.5302 by λ=−2 is actually
  the steeper direction numerically, but the *local* peak sits on the negative
  side, unlike frozen). This is the first reciprocal experiment where adding
  the reciprocal term with the "expected" positive sign actively **hurts**
  the fine-tuned model on holdout.
- **All-200 kept the same sign as frozen** (small positive peak, λ≈0.5), just
  a much smaller gain (+0.0071 vs. frozen's +0.0085 — comparable magnitude,
  same direction). So the sign disagreement is holdout-specific, not a
  wholesale reversal.
- **Given the out-of-distribution caveat above, the most defensible reading
  is: this doesn't yet answer whether fine-tuning helps the reciprocal
  mechanism.** It answers a narrower question — "does a model fine-tuned on
  full-profile text, when repurposed for the look/bg split, keep the earlier
  sweep's positive-λ pattern" — and the answer is *no, not reliably*, with the
  populations disagreeing on sign. Whether that's genuine model behavior or
  an artifact of scoring the fine-tune outside its training distribution is
  unresolved by this run alone.

## What would actually resolve this

Fine-tune `top1_ctrl` (or a fresh adapter) directly on the look/bg split's
text shape, then re-run this sweep — that removes the out-of-distribution
confound and would be the first genuine test of "does fine-tuning change how
much the reciprocal term is worth."

## Repro

```bash
modal run baselines/reciprocal_lambda_grid_top1ctrl/modal_eval.py --run-id real200_001
modal volume get dorby-reciprocal-lambda-grid-top1ctrl-eval real200_001/metrics.json \
    ./artifacts/reciprocal_lambda_grid_top1ctrl/metrics.json

pytest tests/test_reciprocal_lambda_grid_top1ctrl.py -q
```
