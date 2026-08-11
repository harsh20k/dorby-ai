# Does a leash to the frozen model fix the early-overfit pattern?

## Question

Every custom-loop two-tower run in this project so far — `run_001`,
`twotower_split/split_001`, `twotower_field_gate/field_gate_001` — shows the
same shape: dev recall@1 peaks at epoch 1-2, then gets *worse* every epoch
after, despite training loss still falling. That is the textbook signature of
a model overfitting a small dataset (583 training rows) and drifting away
from voyage-4-nano's already-good pretrained query/document alignment.

KL-divergence regularization against a frozen reference model is the
standard fix for exactly this failure mode in other fine-tuning settings
(e.g. RLHF's penalty against a reference policy). This experiment tests it
here: does penalizing the LoRA adapter for drifting away from the frozen
base model's own judgment prevent the early-peak-then-decay pattern, and
does it beat `top1_ctrl` — the project's best two-tower fine-tune — on the
canonical all-200 real-pair scoring?

## Method

New isolated package `twotower_kl_reg/`. Every setting is held identical to
`top1_ctrl_001` (`twotower_top1_optimised/`, currently the best two-tower
fine-tune in this project — `docs/baseline-results-real200.md`):

- **One shared tower**, not two separate ones (the split-tower experiment
  already found separate towers worse).
- **Seeker text = profile + search query concatenated**, the same input
  `top1_ctrl` trained on — not the query-only/no-query variants tried
  elsewhere, since neither beat this representation.
- **Candidate text** = candidate profile only, unchanged from every
  experiment in this project.
- Same rows: `artifacts/twotower_rrf_triplet_ablation/rrf_003_multineg_k1.json`
  (643 rows, 583 train / 60 dev after the seeker-disjoint carve), mounted
  read-only, never copied or regenerated.
- Same LoRA shape (rank 8, alpha 16, dropout 0.05, q/k/v/o_proj — 983,040
  trainable params), same micro-batch 6 / accum 2 (effective batch 12, 245
  optimizer steps), same lr 2e-4, 5 epochs.
- Same main loss: `MultipleNegativesRankingLoss(scale=20.0)` — library
  defaults, no hardness weighting (`top1_ctrl`'s control corner, not the
  sharpened `top1_001` arm that backfired).
- Same checkpoint selection: `CorpusRecallDevEvaluator`, `primary_metric=
  "recall@1"` — ranks each dev anchor against the full 86-candidate dev
  corpus, the fix that made `top1_ctrl` good in the first place.

**The one new thing:** `losses.py::KLRegularizedMNRL` adds a term that
penalizes the LoRA-adapted model's in-batch similarity distribution (the
same `query_to_doc` joint softmax `MultipleNegativesRankingLoss` computes,
`scale=20.0`) for diverging from the *frozen* base model's distribution on
the identical batch:

```
loss = MultipleNegativesRankingLoss(scale=20.0)   # top1_ctrl's exact main term
     + kl_weight * KL(frozen_softmax ‖ adapted_softmax)
```

The frozen distribution is obtained by toggling the *same* PEFT-wrapped
model's adapter off (`model[0].auto_model.disable_adapters()` /
`.enable_adapters()`, from `transformers.integrations.peft.PeftAdapterMixin`)
rather than loading a second model instance — cheap, and guarantees the
"frozen" comparison is against the exact starting weights, not a
separately-loaded copy that could drift on revision/config. `kl_weight=0.5`
is a first, reasoned-but-untuned choice (a follow-up sweep is the natural
next step if this weight moves nothing).

**Verified locally before any GPU spend** (`smoke_kl_backward` in `train.py`,
matching the project-wide convention of a pre-flight gradient check):
1. At LoRA init the adapter-enabled and adapter-disabled forward passes are
   numerically identical (`max_diff_at_init = 0.0`) — expected, since LoRA's
   `B` matrix is zero-initialized, and confirms `disable_adapters()` really
   produces the true frozen output rather than something corrupted.
2. Backward through the combined loss produces non-zero LoRA gradients
   (`lora_grad_sum = 509.22` on the smoke batch).
3. The adapter is left enabled afterward (`active_adapters_after: ['default']`)
   — the toggle inside `forward()` doesn't leak a disabled state into the
   rest of training.

`data.py` and `eval_dev.py` are copies of `twotower_top1_optimised`'s (not
imports, per the isolation rule), pinned byte-for-byte by
`tests/test_kl_reg.py` so this arm cannot silently drift onto a different
population or a different checkpoint-selection rule than `top1_ctrl` used.

```bash
python -m pytest tests/test_kl_reg.py -q

python -m twotower_kl_reg.train --run-id local_smoke \
    --rows-path artifacts/twotower_rrf_triplet_ablation/rrf_003_multineg_k1.json \
    --dry-run --no-run-holdout   # local verification, no GPU

modal run twotower_kl_reg/modal_train.py --run-id kl_reg_ctrl_001 --kl-weight 0.5
modal volume get dorby-twotower-kl-reg-checkpoints kl_reg_ctrl_001 \
    ./artifacts/twotower_kl_reg/kl_reg_ctrl_001

modal run twotower_kl_reg/modal_eval.py --run-id kl_reg_ctrl_001
modal volume get dorby-twotower-kl-reg-eval-results kl_reg_ctrl_001 \
    ./artifacts/twotower_kl_reg/kl_reg_ctrl_001_real200
```

## Training behavior

Dev recall@1 rose monotonically and then held — **no decay**:

| epoch | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| dev recall@1 | 0.2667 | 0.2833 | **0.3333** | 0.3333 | 0.3333 |
| dev MRR | 0.4930 | 0.5009 | 0.5210 | 0.5197 | 0.5169 |

Checkpoint selection kept epoch 3 (first epoch to hit the plateau). Unlike
`run_001`, `split_001`, and `field_gate_001` — which all peaked at epoch 1-2
and got measurably *worse* every epoch after — this run never declines. On
the specific question "does the leash stop the collapse," the answer is yes.

**But a check against `top1_ctrl_001`'s own training curve (pulled for
comparison, not part of the original plan) complicates that reading.**
`top1_ctrl_001` — no KL term at all — already looks like this:

| epoch | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| dev recall@1 (top1_ctrl, no KL) | 0.3000 | 0.3333 | 0.3333 | 0.3167 | 0.3333 |

`top1_ctrl` was already flat, not declining — the sharp early-peak-then-decay
pattern this experiment set out to fix belongs to the *other* architectures
(two independently-initialized towers in `split_001`, an extra learned gate
module in `field_gate_001`), not to `top1_ctrl`'s recipe (one shared tower,
the library's own `MultipleNegativesRankingLoss`, careful recall@1-based
checkpoint selection). That recipe was already stable on its own. The
premise — "top1_ctrl needs a leash to stop it overfitting" — turns out to be
not quite true; what actually needed a leash were the more complex custom
forward passes tried elsewhere, and those have a genuinely different
topology (two models, or a combiner module) that this loss's single-tower
KL mechanism doesn't directly address without further work.

## Results — all 200 real pairs

| Approach | Pair AUC | Hard-neg AUC | Easy-neg AUC | MRR | Recall@1 | Recall@10 |
|---|---|---|---|---|---|---|
| `top1_ctrl` (no KL — best two-tower fine-tune in the project) | **0.5683** | **0.5484** | **0.6836** | **0.3550** | **0.19** | **0.69** |
| `kl_reg_ctrl_001` (this experiment, kl_weight=0.5) | 0.5504 | 0.5148 | 0.6616 | 0.3376 | 0.18 | 0.67 |

**The KL leash is a small net negative, not an improvement.** Every metric
drops slightly — pair AUC −0.018, hard-neg AUC −0.034, MRR −0.017, recall@10
−0.02. Recall@1 (0.18 vs 0.19) is one query out of 100, within noise. None of
this is a large effect, but there is no metric where the leash wins either.

**The holdout misled again** — the now-familiar pattern, seen for the 5th+
time in this project. `kl_reg_ctrl_001`'s holdout looked like the
best-scoring two-tower holdout result yet (pair AUC 0.5724, MRR 0.5242,
recall@1 **0.3793**, hard-neg AUC 0.5586 — all higher than `top1_ctrl`'s own
holdout numbers), which would have read as a clean win if this experiment
had stopped there. Scored on all 200 it reverses to a small loss on every
metric. Standing project rule held again: score on all 200 before calling
anything a result.

## What this means

Two separate questions got asked at once, and they came back with different
answers:

1. **Does a KL-to-frozen-base penalty stop the early-epoch overfit/decay
   pattern seen in this project's more complex custom-loop architectures?**
   Plausibly yes in mechanism — dev recall@1 monotonically improved and
   plateaued rather than declining — but this experiment didn't actually
   test it *on* those architectures (`split_001`, `field_gate_001`), only on
   `top1_ctrl`'s recipe, which turned out not to need the fix in the first
   place. That remains an open question, not a confirmed fix.
2. **Does adding the leash to the project's already-best recipe make it
   better?** No — on the population that actually matters (all 200), it is a
   small, consistent net negative. The most likely explanation: `top1_ctrl`'s
   single shared tower was already well-behaved, so the extra penalty term
   only adds drag against the main task without a collapse to prevent,
   trading a small amount of task performance for stability the recipe
   didn't need.

Combined with `twotower_split/` and `twotower_field_gate/`, this is now the
**third** architectural or loss-level addition tried against `top1_ctrl`'s
plain recipe, and the third to lose on all-200. `top1_ctrl` plus
eval-time query-weighting (`docs/twotower-no-query-experiment.md`, recall@1
0.32 with zero extra training) remains the strongest, cheapest lever found in
this project — reinforced again, not undermined, by this result.

If the underlying hypothesis (a leash prevents overfit collapse) is still
worth testing, the more informative next step is applying it to `split_001`
or `field_gate_001`'s actual training loop — the architectures that showed
the collapse this mechanism was built to fix — rather than to `top1_ctrl`,
which didn't have that problem.

Published artifact: https://claude.ai/code/artifact/62bb2021-7d76-41c3-b05e-3d257df0a236
