# Two-Tower Fine-Tune: Implementation Plan

## Data composition


| Set       | Source                                            | Size | Use                                                                                                      |
| --------- | ------------------------------------------------- | ---- | -------------------------------------------------------------------------------------------------------- |
| Train     | 160 real seed pairs + 500 synth (250 pos/250 neg) | ~660 | Fine-tuning                                                                                              |
| Train-dev | Carve ~20 pairs out of train                      | ~20  | Hyperparameter tuning / early stopping                                                                   |
| Holdout   | Real, frozen, user-disjoint                       | 40   | **Final eval only** — don't peek repeatedly, or you leak into decisions the same way training data would |


Carving a train-dev slice matters: if you tune hyperparameters against the 40-pair holdout across many runs, you're indirectly overfitting to it. Holdout stays for the go/no-go decision only.

## Architecture decision

**Don't train from scratch or full fine-tune** — 660 examples is small. Recommended: **LoRA/adapter fine-tune on top of voyage-4-nano** (open-weight, 340M params, already explored).

- Backbone: `voyageai/voyage-4-nano`, frozen base weights
- LoRA adapters on attention layers (rank 8–16 is typical starting point for this data scale)
- Small trainable projection head per tower if you want asymmetric seeker/candidate transforms beyond just the prompt prefix (optional — start without it, add only if plain LoRA underperforms)
- Keep the existing asymmetric prompt convention (`encode_query` / `encode_document`) — don't discard what already works

This avoids catastrophic forgetting of general retrieval ability while still adapting to your intro-quality signal.

## Loss & batch construction

Use **sentence-transformers'** `SentenceTransformerTrainer` with `MultipleNegativesRankingLoss` (or `CachedMultipleNegativesRankingLoss` if batch size is constrained) — it natively supports:

- (seeker, positive_candidate, hard_negative_candidate) triples
- Combines your labeled hard negatives with in-batch negatives automatically

Batch construction: each training example = seeker text + its labeled hard negative from the same seed/failure_mode, not random pairing — this is exactly why the synth pipeline generated explicit hard negs instead of relying on in-batch-only.

## Training loop


| Step                 | Detail                                                                                                                            |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| Framework            | `sentence-transformers` v3 trainer + `peft` for LoRA                                                                              |
| Epochs               | Few (3–5) — small data, watch train-dev loss for early stopping                                                                   |
| Eval during training | Pair AUC / MRR on train-dev slice only                                                                                            |
| Final eval           | Same metrics on frozen 40-pair holdout — one-time                                                                                 |
| Tracking             | W&B or MLflow for training runs (separate concern from LangSmith, which is for the LLM data-gen pipeline, not embedding training) |




## Eval protocol (unchanged from baseline comparison)

Reuse your existing metrics exactly: pair AUC (overall + hard-neg slice), MRR, Top-1, R@10 — same 40-pair holdout corpus, same protocol as the BERT/nano/large comparison. This is the only way the comparison is meaningful.

**Success bar** (from your earlier plan): beat voyage-4-large — not just voyage-nano or BERT — with a clearer pos/neg cosine gap than the current ~0.01.

## Decision gate

Same rule as the data-gen plan: if hard-neg AUC lift is ≥2–3 points and the gap holds on holdout (not just train-dev), proceed — generate the next data batch (2k) and retrain. If lift shows on train but not holdout, stop and diagnose (likely: hard negatives not hard enough, or LoRA rank/epochs need adjustment) — don't scale data yet.

## Deferred (later phases)

- Precompute candidate embeddings + ANN index for serving
- Cross-encoder reranker on top-K
- MoE/intent-gated routing



## Minimal project layout addition

```
twotower/
  train.py          # SentenceTransformerTrainer + LoRA config
  data.py            # loads real+synth train, train-dev split, holdout
  eval.py             # pair AUC / MRR / R@10 — shared logic with baseline script
  config.py           # LoRA rank, lr, epochs, batch size
```

Want me to write the actual `train.py` skeleton with the LoRA config and loss setup next?