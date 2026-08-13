# Lambda sensitivity sweep on ask_offer_posbg_001

**Status: run complete (`ask_offer_posbg_001` two LoRA towers, voyage-4-nano,
Modal A100, package `reciprocal_lambda_grid_ask_offer_posbg/`).** Same no-fitting
λ sweep as `docs/reciprocal-lambda-grid-experiment.md` and the three later
charts on that page, now on the corrected ask/offer retrain: offer =
`positioning` + `background` only at train *and* eval. Fifth chart on
[`reciprocal-lambda-grid.html`](html/reciprocal-lambda-grid.html).

Chart 4 on that page is the wide-offer `ask_offer_001` sweep (every field
except `lookingFor`). This run does not replace it.

## What changed vs chart 4

New isolated package **`reciprocal_lambda_grid_ask_offer_posbg/`** (neither
`reciprocal_lambda_grid_ask_offer/` nor `twotower_ask_offer_posbg/` edited).
Adapters from `ask_offer_posbg_001` / volume
`dorby-twotower-ask-offer-posbg-checkpoints`. Offer text from
`twotower_ask_offer_posbg.text` (`BG_FIELDS = ("positioning", "background")`),
not `baselines.reciprocal_static.text`. Ask packing unchanged. No fitting: λ
swept from −2 to +2 (step 0.05, 81 points), pair ROC-AUC of
`s_fwd + λ·s_recip` reported directly against the labels being scored.

The model itself was trained at a fixed λ=1.75. This sweep is a post-hoc
diagnostic on top of that, not a value chosen for deployment.

## Results

| population | forward-only AUC (λ=0) | at trained λ=1.75 | curve max |
|---|---|---|---|
| all 200 (headline) | 0.5159 | **0.5754** | λ=2.00 → 0.5782 (grid boundary, not a peak) |
| holdout (69 pairs, not headline) | 0.5853 | 0.5983 | λ=0.90 → 0.6060 (small genuine bump) |

λ=1.75 all-200 combined 0.5754 matches the already-published fixed-λ eval
(`docs/twotower-ask-offer-posbg-experiment.md`) to four decimals.

Curve shape (selected λ, full 0.05-step grid in
`artifacts/reciprocal_lambda_grid_ask_offer_posbg/metrics.json`):

| λ | holdout AUC | all-200 AUC |
|---|---|---|
| -2.00 | 0.4690 | 0.4576 |
| 0.00 (forward-only) | 0.5853 | 0.5159 |
| +0.90 (holdout peak) | **0.6060** | 0.5559 |
| +1.75 (trained) | 0.5983 | **0.5754** |
| +2.00 (grid edge) | 0.5948 | **0.5782** |

## Reading it against chart 4 (wide-offer ask_offer_001)

Narrowing offer to positioning+background does not change the λ-sensitivity
story. All-200 at the trained λ is 0.5754 vs wide-offer 0.5714 — a wash.
Forward-only is still near chance (0.5159 vs 0.5126). The reciprocal term
still supplies the lift (+0.059). All-200 still has no interior peak: it
rises to the grid boundary (here λ=2.00 → 0.5782; chart 4 rose to λ=1.90 →
0.5723), a shelf, not a chosen λ.

Holdout peaks later and lower than chart 4 (λ=0.90 → 0.6060 vs λ=0.15 →
0.6293). That split is not the ranking among strong models.

Every point is read off the same labels being scored, so “best λ” is
optimistic by construction. This does not validate a deployable λ. Still
does not beat `voyage_gemini_ctrl` combined all-200 (0.6587).

## Reproduce

```bash
python -m pytest tests/test_reciprocal_lambda_grid_ask_offer_posbg.py -q
modal run reciprocal_lambda_grid_ask_offer_posbg/modal_eval.py --run-id ask_offer_posbg_001
modal volume get dorby-reciprocal-lambda-grid-ask-offer-posbg-eval \
    ask_offer_posbg_001/metrics.json \
    ./artifacts/reciprocal_lambda_grid_ask_offer_posbg/metrics.json
```
