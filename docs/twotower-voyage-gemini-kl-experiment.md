# The KL leash loses again, on the bigger batch too — no, it doesn't help

## Question

`twotower_kl_reg` tested a KL-divergence-against-frozen-base penalty
(`kl_weight=0.5`) on `top1_ctrl`'s recipe and `rrf_003` — 643 rows, 583
train / 60 dev, a small and comparatively clean synthetic batch — and found
a small net negative on all 200 real pairs: pair AUC 0.5504 vs `top1_ctrl`'s
0.5683, every metric slightly worse
(`docs/twotower-kl-reg-experiment.md`). That negative result was measured on
one small dataset. This experiment retests the identical KL mechanism on
`voyage_gemini_ctrl_001`'s recipe and its much bigger, differently-sourced,
measurably leakier training population
(`pairing_voyage_gemini/smoke_test_002`, 3,008 rows / 1,921 seekers, 4.7x
`rrf_003`'s row count) to see whether the earlier negative verdict
generalizes or was specific to the small-data regime.

## Method

New isolated package `twotower_voyage_gemini_kl/`. Two things combined,
each copied from its own source rather than reimplemented:

- **Recipe + data**, copied from `twotower_voyage_gemini_ctrl/`: LoRA rank 8
  / alpha 16 / dropout 0.05 on q/k/v/o_proj (983,040 trainable params,
  confirmed on the real run), micro-batch 6 / accum 2 (effective batch 12),
  lr 2e-4, 5 epochs, `voyage-4-nano` at native 1024-dim truncation,
  `CorpusRecallDevEvaluator` recall@1 checkpoint selection, full-profile text
  on both sides. Training rows: `artifacts/twotower_voyage_gemini_ctrl/
  voyage_gemini_smoke002_multineg_k1.json` (3,008 rows / 1,921 seekers / 0%
  padding), mounted read-only from that package's artifacts dir — never
  copied or regenerated, matching how `twotower_kl_reg` itself mounted
  `rrf_003_multineg_k1.json` read-only from `twotower_rrf_triplet_ablation`.
- **Loss mechanism**, copied verbatim from `twotower_kl_reg/losses.py`:
  `KLRegularizedMNRL` — `MultipleNegativesRankingLoss(scale=20.0)` (the
  library's own, `top1_ctrl`'s/`voyage_gemini_ctrl_001`'s exact main term)
  plus `kl_weight=0.5 * KL(frozen_softmax ‖ adapted_softmax)`, computed over
  the identical `query_to_doc` joint softmax the main loss uses. The frozen
  distribution comes from toggling the same PEFT model's adapter off
  (`disable_adapters()`/`enable_adapters()`), not a second model instance.
  `kl_weight=0.5` is unchanged from the prior experiment — this is a true
  retest of the same idea, not a re-tuned one.

`data.py`/`eval_dev.py` are byte-identical copies (after the module
docstring) of `twotower_voyage_gemini_ctrl`'s; `losses.py` is a
byte-identical copy of `twotower_kl_reg`'s. All three are pinned by
`tests/test_voyage_gemini_kl.py` so the copies cannot silently drift from
either source experiment. Nothing under `twotower_voyage_gemini_ctrl/`,
`twotower_kl_reg/`, or `twotower/` was modified.

**Verified locally before any GPU spend** (`--dry-run --no-run-holdout`):
LoRA target counts correct (12 per target, 983,040 trainable params), row
counts as expected (2,710 train / 298 dev), and the KL smoke check
(`smoke_kl_backward`) passed — `max_diff_at_init=0.0` (adapter-enabled and
adapter-disabled forward passes identical at LoRA init, confirming
`disable_adapters()` produces the true frozen output) and nonzero LoRA
gradients through the combined loss. The same check re-ran automatically on
the real Modal run and passed again (`lora_grad_sum=504.98`).

```bash
python -m pytest tests/test_voyage_gemini_kl.py -q

python -m twotower_voyage_gemini_kl.train --run-id local_smoke \
    --rows-path artifacts/twotower_voyage_gemini_ctrl/voyage_gemini_smoke002_multineg_k1.json \
    --dry-run --no-run-holdout   # local verification, no GPU

modal run --detach twotower_voyage_gemini_kl/modal_train.py --run-id voyage_gemini_kl_001
modal volume get dorby-twotower-voyage-gemini-kl-checkpoints voyage_gemini_kl_001 \
    ./artifacts/twotower_voyage_gemini_kl/voyage_gemini_kl_001
# (pull run_meta.json/run_result.json/metrics_holdout.json/loss_history.json and
#  the adapter/ subfiles individually — `modal volume get` on a whole run dir
#  errors "Is a directory" on this CLI version)

# holdout-only sanity check, computed on Modal (not locally) — see Caveats
modal run twotower_voyage_gemini_kl/modal_eval.py --run-id voyage_gemini_kl_001
```

## Training behavior

Dev recall@1 rose through epoch 3, plateaued, then dipped slightly at epoch 5:

| epoch | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| dev recall@1 | 0.4128 | 0.4228 | **0.4329** | 0.4329 | 0.4161 |
| dev MRR | 0.6176 | 0.6262 | 0.6332 | 0.6250 | 0.6155 |

Checkpoint selection kept the first checkpoint to reach the peak score
(epoch 3, step 678 — epoch 4 tied the same score but strict-greater
selection keeps the earlier one). Final train loss 0.6326 (mean 0.7308 over
the run). This is a much milder version of the early-peak-then-decay pattern
`twotower_kl_reg`'s motivating cases (`run_001`, `split_001`,
`field_gate_001`) showed — a real but small dip (0.4329 → 0.4161) rather
than a collapse, on a recipe (`voyage_gemini_ctrl`'s, which is `top1_ctrl`'s)
that `twotower_kl_reg`'s own doc already found doesn't need a leash in the
first place.

## Results — all 200 real pairs (the population that decides this)

Registered in `eval_real_full/guard.py`'s `SYNTHETIC_ONLY_ROW_SOURCES`
(already covered — this run reuses `voyage_gemini_ctrl_001`'s rows file
unchanged, so no allowlist edit was needed) and `eval_real_full/modal_eval.py`'s
`CONFIGS` dict (`voyage_gemini_kl`, purely additive — no existing entry
touched), then scored via `eval_real_full.eval.run_eval`, unmodified, the
same shared path every other all-200 number in this project goes through.

| Approach (all 200 real pairs, corpus 178) | Pair AUC | Hard-neg AUC | MRR | Recall@1 | Recall@10 |
|---|---|---|---|---|---|
| `voyage_gemini_ctrl_001` (no KL) | **0.6081** | **0.6264** | **0.4506** | **0.26** | **0.81** |
| `voyage_gemini_kl_001` (this run, kl_weight=0.5) | 0.5479 | 0.5230 | 0.3998 | 0.22 | 0.77 |

**The KL leash is a clear net negative on all 200, same as it was on the
small batch.** Every metric drops: pair AUC −0.0602, hard-neg AUC −0.1034
(the largest relative drop of any metric — the leash hurts hard-negative
discrimination most), MRR −0.0508, recall@1 −0.04 (4 fewer correct top-1
matches out of 100), recall@10 −0.04. This is not a small-margin,
within-noise result — it is worse on every metric by a wide margin, wider
than the small-batch `kl_reg_ctrl_001` result was (pair AUC there dropped
0.018, here 0.060).

**Unlike most custom-loop experiments in this project, the holdout did not
mislead this time — it just understated the size of the loss.** Both
populations agree on direction:

| Population | Pair AUC (no KL → KL) | Hard-neg AUC (no KL → KL) | MRR (no KL → KL) |
|---|---|---|---|
| Holdout (69 pairs) | 0.6802 → 0.6147 (−0.0655) | 0.7431 → 0.6638 (−0.0793) | 0.6585 → 0.5862 (−0.0723) |
| All 200 (this table) | 0.6081 → 0.5479 (−0.0602) | 0.6264 → 0.5230 (−0.1034) | 0.4506 → 0.3998 (−0.0508) |

Direction matches on every metric; magnitude is similar on pair AUC and MRR,
but the all-200 population shows a noticeably larger hard-negative AUC drop
(−0.1034 vs holdout's −0.0793) — the leash hurts the metric that matters
most for this project's real deployment population (production's own false
positives) more than the smaller holdout sample suggested.

## Caveats

- **Labels are an unreviewed pilot batch's LLM-judge opinions**, not real
  accept/decline outcomes. `pairing_voyage_gemini/smoke_test_002` is
  named as a pipeline pilot, not a promoted or human-reviewed dataset, and
  measured leakier than `rrf_003` *before any training happened*: basic
  leakage checks found candidate-profile-only AUC 0.758, seeker-identity-only
  AUC 0.780, and 43.5% of seekers all-positive or all-negative across every
  query they asked — all three worse than `rrf_002`, the batch `rrf_003`
  (and therefore `kl_reg_ctrl_001`) traces back to. Lexical circularity
  stayed clean (TF-IDF query-candidate cosine AUC 0.481, near chance) — this
  batch is not a keyword-overlap shortcut, but it is a bigger base-rate/
  candidate-identity shortcut than anything the KL mechanism was previously
  tested against. See `scripts/leakage_check_pairing_voyage_gemini.py` and
  `docs/twotower-voyage-gemini-ctrl-experiment.md`'s own caveats section for
  the full framing — this arm inherits that caveat unchanged, since it
  trains on the identical rows.
- **A step of this run's holdout eval briefly ran locally before being
  redone on Modal.** Mid-run, `twotower_voyage_gemini_kl/eval.py` was
  invoked directly on the local machine and got partway through computing
  embeddings (candidate-corpus encoding) before being killed and re-run as
  `modal_eval.py` on Modal GPU instead, per this project's "compute on
  Modal, pull only small result files" convention. The final numbers
  reported here are from the Modal run only; the aborted local run produced
  no output and is not reflected in any file.
- **This is a same-mechanism, different-data comparison only.** It answers
  "does the KL leash's small-batch verdict generalize to a bigger batch,"
  not "is this batch or this mechanism worth deploying" — see
  `twotower_voyage_gemini_ctrl`'s own caveats for the separate,
  already-answered question of whether `top1_ctrl`'s plain recipe transfers
  to this data (it does, and beats `top1_ctrl` on all-200: pair AUC 0.6081
  vs 0.5683).

## What this means

The small-batch verdict generalizes: the KL leash is a net negative on
`voyage_gemini_ctrl_001`'s recipe, on both the small `rrf_003` batch
(`kl_reg_ctrl_001`) and this much bigger, leakier one — and the loss is
larger here, not smaller. Mechanically the KL term works exactly as
designed (zero divergence at LoRA init, nonzero gradients through the
combined loss, adapter correctly toggled, a dev recall@1 curve that peaks
and plateaus rather than collapsing) — the mechanism isn't broken, it just
doesn't help this recipe. `twotower_kl_reg`'s own framing was that the leash
targets architectures with a sharp early-peak-then-collapse pattern
(`split_001`, `field_gate_001`); `top1_ctrl`'s plain recipe (which both
`voyage_gemini_ctrl_001` and this run inherit) never showed that sharp a
collapse to begin with — its dev recall@1 only dipped mildly, 0.4329 →
0.4161, the mildest version of the pattern in the project — so there was
never much for the leash to fix here. Two-for-two against this recipe now;
the leash remains untested against an architecture that actually collapses
sharply, which is the population it was designed for.

Checkpoint: `artifacts/twotower_voyage_gemini_kl/voyage_gemini_kl_001/adapter/`
(also on the `dorby-twotower-voyage-gemini-kl-checkpoints` Modal volume).
All-200 metrics: `artifacts/eval_real_full/real200_voyage_gemini_kl/voyage_gemini_kl/metrics.json`.
