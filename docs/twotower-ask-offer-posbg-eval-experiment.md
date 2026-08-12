# ask_offer_001 re-eval: offer = positioning + background

Isolated package: `twotower_ask_offer_posbg_eval/`.  
Does **not** edit `twotower_ask_offer/`. Does **not** retrain.

## Question

If we keep the trained `ask_offer_001` adapters and only change offer-side
text at eval to the fields the design specified (`positioning` +
`background`), what happens on the **200 real pairs**?

This is an eval-time field swap. The offer tower was still trained on every
profile field except `lookingFor`.

## Setup

| Item | Value |
|---|---|
| Adapters | `ask_offer_001` (Modal volume, read-only) |
| Population | all 200 real pairs (100 pos / 100 neg, 178 candidates) |
| Holdout split | not scored |
| Ask text | `lookingFor` + seeker `searchQuery` (unchanged) |
| Offer text | `positioning` + `background` only |
| λ | 1.75 (training value, not refit) |

## All-200 results (2026-08-12)

| Metric | This run (pos+bg offer) | Original `ask_offer_001` (wide offer) |
|---|---:|---:|
| Pair AUC forward-only | 0.5206 | 0.5126 |
| Pair AUC combined | **0.5572** | **0.5714** |
| Hard-neg AUC (combined) | 0.5464 | 0.5236 |
| Easy-neg AUC (combined) | 0.6396 | 0.6952 |
| MRR (forward retrieval) | 0.4221 | 0.4317 |
| Recall@1 | 0.2600 | 0.2700 |

Artifact: `artifacts/twotower_ask_offer_posbg_eval/all200/metrics.json`.

Published findings page:
[https://dorby-project-story-411960113601.s3.amazonaws.com/docs/twotower-ask-offer-posbg-eval.html](https://dorby-project-story-411960113601.s3.amazonaws.com/docs/twotower-ask-offer-posbg-eval.html)
(local: `docs/html/twotower-ask-offer-posbg-eval.html`).

## Reading

Narrowing offer text at eval **does not recover** the intended experiment.
Combined pair AUC drops 0.5714 → 0.5572. Forward-only ticks up slightly
(0.5126 → 0.5206). Retrieval is a hair worse.

The reciprocal term still helps vs forward-only (0.5206 → 0.5572), just less
than with the wider offer text the tower actually trained on.

A correct test of the design needs a **new training run** with
positioning+background offer text, not this swap.

## Reproduce

```bash
python -m pytest tests/test_twotower_ask_offer_posbg_eval.py -q
modal run twotower_ask_offer_posbg_eval/modal_eval.py --run-id ask_offer_001 --lam 1.75
modal volume get dorby-twotower-ask-offer-posbg-eval all200 \
  ./artifacts/twotower_ask_offer_posbg_eval
```
