# ask/offer retrain: offer = positioning + background

Isolated package: `twotower_ask_offer_posbg/`.  
Does **not** edit `twotower_ask_offer/`.

Corrected training run of the ask/offer design: Ask = `lookingFor` (+
seeker `searchQuery`); Offer = `positioning` + `background` only. Same 3008
frozen rows, same λ=1.75, same LoRA recipe as `ask_offer_001`.

Status: training launched 2026-08-12 (`ask_offer_posbg_001`). Results TBD.

```bash
python -m pytest tests/test_twotower_ask_offer_posbg.py -q
modal run twotower_ask_offer_posbg/modal_train.py --run-id ask_offer_posbg_001 --lam 1.75 --epochs 5
```
