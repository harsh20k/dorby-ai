# A learned gate over query + `lookingFor` + `positioning`

## Question

`query_weighted/` found that a *fixed* alpha blend (profile vs. query,
picked once for the whole project) beats plain concatenation. That leaves
an open question flagged after `twotower_no_query/`/`twotower_query_only/`
ruled out "training on one whole-text variant vs. another": would a
**learned, per-seeker** combination beat a fixed global blend — and could
decomposing the *profile itself* into fields (not just profile-vs-query)
give the model something more useful to combine?

This experiment: one shared tower (unlike `twotower_split/`, which found
two independent towers hurt), seeker side split into three pieces —
`query_only` text, `lookingFor`, `positioning` — each encoded separately,
combined by a small trainable gate (`gate.py::FieldGate`, 9,219 params: one
linear layer over the concatenated piece embeddings, softmax to per-piece
weights, weighted sum) instead of one fixed alpha.

## Method

New isolated package `twotower_field_gate/`. Same base model and LoRA shape
as every sibling experiment (rank 8/alpha 16/dropout 0.05, q/k/v/o_proj),
same 5-epoch/lr-2e-4 budget, same rows source
(`scripts/build_rrf_multineg_triplets_field_pieces.py`, a near-copy of the
query-only row script that keeps the three seeker pieces separate instead
of concatenating them — verified 0/643 rows needed the missing-field
fallback). Candidate side unchanged (`candidate_to_text`); the novelty is
scoped to the seeker side only, per the design question.

Custom training loop for the same reason as `twotower_split/`
(`SentenceTransformerTrainer` can't express "encode three pieces, run them
through an extra trainable module, then contrast"). Verified before any GPU
spend: both the LoRA adapter and the gate receive independent nonzero
gradients on a smoke batch (LoRA grad sum 236.6, gate grad sum 16.9).

**A real bug found and fixed while building this**, worth stating plainly:
the custom loop's raw `model(features)["sentence_embedding"]` call returns
voyage-4-nano's *native* 2048-dimension embedding, not the 1024-dim
truncated one every other number in this project is computed on —
`truncate_dim=1024` is only applied inside `SentenceTransformer.encode()`'s
own post-processing (`sentence_transformers.util.truncate_embeddings`,
literally `embeddings[..., :truncate_dim]`, applied to the already-normalized
pooled output, then renormalized). Confirmed numerically before wiring the
fix in: manually slicing to 1024 dims and renormalizing exactly reproduces
`.encode(..., truncate_dim=1024, normalize_embeddings=True)`'s output
(max abs diff 0.0). `twotower_split/train.py` has this same untruncated-
forward-pass pattern and was **not** retroactively fixed or rerun — its
published numbers stand as computed, with this now on record as a caveat:
it trained on the native 2048-dim space and was only truncated to 1024 at
eval time, which this package avoids by truncating consistently everywhere,
train and eval alike.

```bash
python scripts/build_rrf_multineg_triplets_field_pieces.py \
    --batch-dir exports/rrf_datasets/rrf_003 --negatives-per-anchor 1 \
    --out artifacts/twotower_field_gate/rrf_003_multineg_k1_field_pieces.json

modal run --detach twotower_field_gate/modal_train.py --run-id field_gate_001
modal volume get dorby-twotower-field-gate-checkpoints field_gate_001 \
    ./artifacts/twotower_field_gate/field_gate_001

modal run twotower_field_gate/modal_eval.py --run-id field_gate_001
modal volume get dorby-twotower-field-gate-eval-results field_gate_001 \
    ./artifacts/twotower_field_gate/field_gate_001_real200
```

## Training behavior

Dev recall@1 peaked at **epoch 1 (0.3333)** — the highest epoch-1 peak of
any two-tower run in this project — then fell for two epochs, partially
recovered at epoch 3 (0.30), then fell again (epoch 4: 0.20, epoch 5:
0.23). Noisier than either single-model run, but the same overall shape:
early peak, no later epoch beats it. Checkpoint selection kept epoch 1.

## Results — all 200 real pairs

| Approach | Pair AUC | Hard-neg AUC | MRR | Recall@1 | Recall@10 |
|---|---|---|---|---|---|
| `top1_ctrl` + eval-time query-only swap (no retrain) | 0.5945 | 0.6456 | **0.5076** | **0.32** | **0.90** |
| `query_only_001` (single model, trained on query only) | 0.5952 | **0.6492** | 0.4985 | 0.29 | 0.91 |
| `split_001` (two separate towers) | 0.5677 | 0.4832 | 0.4844 | 0.28 | 0.88 |
| **`field_gate_001` (learned gate, 3 pieces)** | 0.5919 | 0.5520 | 0.4204 | 0.23 | 0.78 |

**Mixed, and net negative.** The learned gate beats the split-tower
approach on pair classification (AUC 0.5919 vs 0.5677, hard-neg AUC 0.5520
vs 0.4832) but is the **worst of all four on every retrieval metric** — MRR,
recall@1, and recall@10 all land below every other approach, including the
split towers. Decomposing the profile into pieces and learning to combine
them did not recover, let alone beat, the simplicity of encoding one
coherent `query_only` string.

The holdout again told a flattering, misleading story: R@1 0.5172, MRR
0.6485 — the best-looking holdout number of any two-tower model in this
project — reversed on all 200 to the worst retrieval result of the four.

## What this means

Three architectural variants have now been tried against the simple
single-representation baseline, and all three lost on the metrics that
matter most (MRR/recall):

1. Two separate towers (`twotower_split/`) — worst on hard-neg AUC.
2. A learned per-seeker combination over decomposed profile pieces (this
   package) — worst on MRR/recall@1/recall@10.
3. (Reference) One tower, one fixed representation chosen at eval time
   (`query_only` alone, or `alpha_0.6`) — still the best result in the
   project on every population tested, cheapest to build, and unlike either
   architectural variant, needs zero extra trainable parameters or GPU
   training at all.

The likely common cause across both negative results: extra learnable
structure (a second tower's full weight set, or even just 9,219 gate
parameters) needs more than 583 training rows to reliably beat a simpler
baseline, and at this project's real data scale it doesn't get there. A
single coherent string handed to an already-well-trained frozen encoder
continues to outperform every attempt so far to out-engineer it with more
architecture. The practical recommendation from `docs/twotower-split-
experiment.md` stands, reinforced by a second independent negative result:
the query-weighting trick on a normally-trained shared model — especially
voyage-4-large's query-only arm (recall@1 0.42,
`docs/query-weighted-encoding-experiment.md`) — remains the strongest,
cheapest lever found in this project.

Published artifact: https://dorby-project-story-411960113601.s3.amazonaws.com/docs/twotower-field-gate-experiment.html
