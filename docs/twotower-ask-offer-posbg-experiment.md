# ask/offer retrain: offer = positioning + background

Isolated package: `twotower_ask_offer_posbg/`.
Does **not** edit `twotower_ask_offer/`.

Corrected training run of the ask/offer design: Ask = `lookingFor` (+
seeker `searchQuery`); Offer = `positioning` + `background` only. Same 3008
frozen rows, same λ=1.75, same LoRA recipe as `ask_offer_001`. Train and eval
use the same packing.

**Status: run complete (`ask_offer_posbg_001`, voyage-4-nano, Modal A100).**
All-200 combined pair AUC **0.5754** vs original wide-offer **0.5714** — a
wash. The earlier eval-time swap (0.5572) was train/eval mismatch, not
evidence that the intended field set is worse.

## Question

If we actually *train* the offer tower on positioning + background only
(the design spec), then score all 200 real pairs with those same fields,
does the intended split beat the original wide-offer run?

## Setup

| Item | Value |
|---|---|
| Run | `ask_offer_posbg_001` |
| Package | `twotower_ask_offer_posbg/` (local `text.py`, `BG_FIELDS = ("positioning", "background")`) |
| Rows | same 3008 frozen triplets as `ask_offer_001` / `voyage_gemini_ctrl_001` |
| Ask text | `lookingFor` + seeker `searchQuery` |
| Offer text | `positioning` + `background` only (train *and* eval) |
| λ | 1.75 (fixed, not learned) |
| Best ckpt | epoch 2 (dev R@1 0.3523). Epochs 1–5: 0.322 / **0.352** / 0.342 / 0.336 / 0.336 |
| Headline population | all 200 real pairs (100 pos / 100 neg, 178 candidates) |

## All-200 results (2026-08-12)

| Metric | Retrain pos+bg (this) | Eval-time swap | Original wide offer |
|---|---:|---:|---:|
| Pair AUC forward-only | 0.5159 | 0.5206 | 0.5126 |
| Pair AUC combined | **0.5754** | 0.5572 | **0.5714** |
| Hard-neg AUC (combined) | 0.5444 | 0.5464 | 0.5236 |
| Easy-neg AUC (combined) | 0.6508 | 0.6396 | 0.6952 |
| MRR (forward retrieval) | 0.4231 | 0.4221 | 0.4317 |
| Recall@1 | 0.2400 | 0.2600 | 0.2700 |

Eval-time swap = `ask_offer_001` adapters rescored with pos+bg offer text
(`docs/twotower-ask-offer-posbg-eval-experiment.md`). Original = train+eval
on every profile field except `lookingFor`.

Holdout (69 pairs, not the ranking among strong models): combined 0.5983,
forward 0.5853, R@1 0.3448, MRR 0.5636, hard-neg 0.6328.

Artifact: `artifacts/twotower_ask_offer_posbg/metrics_full_eval.json`.

Published findings page:
[https://dorby-project-story-411960113601.s3.amazonaws.com/docs/twotower-ask-offer-posbg-experiment.html](https://dorby-project-story-411960113601.s3.amazonaws.com/docs/twotower-ask-offer-posbg-experiment.html)
(local: `docs/html/twotower-ask-offer-posbg-experiment.html`).

## Reading

Matching train and eval fields **recovers** the original combined AUC
(0.5572 → 0.5754). Narrowing offer to positioning + background at training
time neither helps nor hurts in a way that would change a decision: +0.004
combined AUC, −0.03 R@1.

The reciprocal term still does the same work as before (forward 0.516 →
combined 0.575, +0.059). Forward-only is still near chance and still below
frozen Voyage-4-nano’s own forward score (0.5638). This design still does
not beat `voyage_gemini_ctrl` combined all-200 (0.6587).

## Reproduce

```bash
python -m pytest tests/test_twotower_ask_offer_posbg.py -q
modal run twotower_ask_offer_posbg/modal_train.py --run-id ask_offer_posbg_001 --lam 1.75 --epochs 5
modal run twotower_ask_offer_posbg/modal_eval.py --run-id ask_offer_posbg_001 --lam 1.75
```
