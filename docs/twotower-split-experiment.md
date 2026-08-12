# A genuinely split two-tower — separate query and candidate adapters

## Question

Every two-tower experiment in this project so far — `top1_ctrl`,
`twotower_no_query/`, `twotower_query_only/` — used **one shared model** for
both the seeker side and the candidate side. The only thing that differed
between roles was which text got fed in (and a role prompt). This experiment
tests the actual architectural idea: two **separate** LoRA adapters — a
dedicated query tower and a dedicated candidate tower — trained jointly, so
each can specialize for its own job instead of sharing one set of weights.

## Method

New isolated package `twotower_split/`. Same base model (`voyage-4-nano`),
same LoRA shape (rank 8, alpha 16, dropout 0.05, q/k/v/o_proj), same rows as
`twotower_query_only/` (`rrf_003_multineg_k1_query_only.json` — reused
directly, not regenerated), same 5-epoch/lr-2e-4 budget. The only real change
is architecture: two independently-initialized adapters instead of one,
built by calling `twotower.train.build_model`/`add_lora_adapter` twice.

`SentenceTransformerTrainer` can't route different dataset columns through
different models, so `train.py` is a from-scratch training loop: manual
batching, a hand-written MultipleNegativesRankingLoss-equivalent (in-batch +
one explicit hard negative, cross-entropy over cosine similarity scaled by
20 — the same formula the library loss uses), and per-epoch dev evaluation
(`baselines.metrics.retrieval_metrics`, reused unchanged) that saves whichever
epoch's adapter pair scores best on dev recall@1.

**Caveat added 2026-08-04** (`docs/possible-bugs.md` #6, found while building
the follow-up `twotower_field_gate/` experiment): this custom loop's raw
`model(features)["sentence_embedding"]` call returns nano's native
2048-dim embedding, not the 1024-dim truncated one every other number in
this project uses (`truncate_dim` is only applied inside
`SentenceTransformer.encode()`'s own post-processing, invisible to a raw
forward pass). So this run trained on the untruncated 2048-dim space and was
only truncated to 1024 at eval time — not retroactively fixed or rerun,
recorded here as a methodological gap rather than left unstated. The
negative finding below may or may not hold at the correct dimensionality;
`twotower_field_gate/` was built with the fix applied from the start.

Verified before any GPU spend: a local dry-run confirmed both towers get
non-zero, independent LoRA gradients on the same smoke batch (query-tower
grad sum 120.9, doc-tower grad sum 170.1 — both nonzero, proving the loss
routes real signal to both sets of weights, not just one).

Evaluation matches training exactly by construction — the query tower only
ever sees `searchQuery` text, the candidate tower only ever sees
`candidate_to_text`, on both sides, everywhere. No eval/train mismatch is
possible here the way it was in `twotower_no_query/`'s first pass
(`docs/possible-bugs.md` #5).

```bash
modal run --detach twotower_split/modal_train.py --run-id split_001
modal volume get dorby-twotower-split-checkpoints split_001 \
    ./artifacts/twotower_split/split_001

modal run twotower_split/modal_eval.py --run-id split_001
modal volume get dorby-twotower-split-eval-results split_001 \
    ./artifacts/twotower_split/split_001_real200
```

## Training behavior

Dev recall@1 **peaked at epoch 1 (0.2833) and declined every epoch after**
(epoch 3: 0.20, epoch 4: 0.13, epoch 5: 0.20) despite training loss
continuing to fall (0.54 -> 0.38 -> 0.33 mean per-epoch loss) — the
checkpoint-selection mechanism caught this and correctly kept epoch 1's
adapters rather than the final ones. This is the same failure shape seen
before in this project (`run_001`, `docs/possible-bugs.md` #2): the model
overfits before training loss stops improving, and it happened *faster* here
than in any single-shared-model run.

## Results — all 200 real pairs

| Approach | Pair AUC | Hard-neg AUC | MRR | Recall@1 | Recall@10 |
|---|---|---|---|---|---|
| `top1_ctrl` + eval-time query-only swap (no retrain, one model) | 0.5945 | 0.6456 | 0.5076 | 0.32 | 0.90 |
| `query_only_001` (one model, trained on query only) | 0.5952 | 0.6492 | 0.4985 | 0.29 | 0.91 |
| **`split_001` (two separate towers, trained jointly)** | **0.5677** | **0.4832** | **0.4844** | **0.28** | **0.88** |

**The split architecture is the worst of the three — clearly, not just
within noise.** Pair AUC drops, MRR drops, and hard-negative AUC drops
sharply to 0.4832, *below chance* — the one metric where every single-model
approach in this project has scored comfortably above 0.59. Recall@1/10 are
close to the other two but still the lowest.

As with `no_query_001`, the holdout number told a different, misleading
story (AUC 0.6267, R@1 0.4483, MRR 0.6506 — the best-looking holdout number
of any two-tower model in the project) that completely reverses on all 200.
Yet another instance of the standing project rule: score on all 200 before
calling anything a result.

## What this means

The prediction from the plain-language writeup ("no guarantee it wins... a
different kind of specialization, not a proven one") landed on the negative
side. The most likely explanation: `voyage-4-nano`'s frozen base already
provides a well-aligned query/document space — Voyage trained that alignment
at a scale this project cannot match, and every prior LoRA adapter here only
nudges that shared space slightly, so the alignment survives training by
construction. Splitting into two independently-initialized adapters throws
that alignment away and asks the model to relearn compatibility between two
specialized subspaces from only 583 training rows — nowhere near enough
data to do that reliably, which is consistent with the faster overfitting
observed above (peak at epoch 1, not epoch 2 like every single-model run).

This doesn't rule out a genuinely split two-tower working with **enough**
data — sizes more like the millions of pairs production two-tower systems
usually train on — but on this project's real-scale rows (583 train / 60
dev), it is a clear net negative versus keeping one shared model and simply
choosing what text to feed it. Combined with `twotower_no_query/` and
`twotower_query_only/`'s findings, the practical recommendation for this
project stands: the query-weighting trick on a normally-trained shared model
(`docs/query-weighted-encoding-experiment.md`, voyage-4-large's 0.42 R@1
extension) remains the best lever found, cheaper and stronger than every
training-architecture variant tried against it.

Published artifact: https://dorby-project-story-411960113601.s3.amazonaws.com/docs/twotower-split-experiment.html
