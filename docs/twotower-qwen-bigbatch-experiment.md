# Qwen3-8B at a real micro-batch: does the nano finding transfer?

`twotower_rrf_triplet_ablation` established, on voyage-4-nano at fixed effective
batch, that **micro-batch size is the lever that moves retrieval** — raising it
2 → 6 gained +0.052 MRR and +0.069 recall@1, while a second negative per row
hurt. The mechanism is `MultipleNegativesRankingLoss`'s in-batch negatives,
which scale with the *true* micro-batch, not the gradient-accumulated effective
batch.

Qwen3-Embedding-8B has the highest pair AUC of any model measured in this
project (frozen 0.6595, above Voyage-4-large's 0.6086). Its one fine-tune,
`twotower_rrf_triplet/rrf_triplet_qwen3_8b_h100_002`, ran at
`train_batch_size=1` — exactly the starved corner the ablation later showed to
be worst — and gained almost nothing: 0.6595 → 0.6672 pair AUC, inside the
measured ±0.013 noise band, with hard-negative AUC and recall@10 both going
*down*.

This experiment asks whether that flat result was the model or the batch size.

Isolated package `twotower_qwen_bigbatch/`. Nothing under `twotower/`,
`twotower_rrf_triplet/`, `twotower_rrf_triplet_bigbatch/`, or
`twotower_rrf_triplet_ablation/` was modified.

## Finding 1 — micro-batch 1 was never a measured ceiling

The prior preset justified `train_batch_size=1` with an OOM observed at **fp32
on a 40GB A100**. That is a weights-memory problem, and the same preset already
solved it twice over: `torch_dtype=bfloat16` (16GB resident instead of 32GB) and
`gradient_checkpointing_override=True`. The ceiling was never re-probed after
those landed, and never on an 80GB card, where the binding constraint is
activations rather than weights.

`probe_batch_size.py` measured it — one real forward+backward per size, same
model, same LoRA config, same loss, real text from the rrf_003 k=1 rows:

| micro-batch | peak memory | of 79.18 GB |
|---|---|---|
| 1 | 15.71 GB | 20% |
| 2 | 17.26 GB | 22% |
| 4 | 20.45 GB | 26% |
| 6 | 23.72 GB | 30% |
| 8 | 27.46 GB | 35% |

**Nothing OOM'd.** Micro-batch 1 used 15.71 GB — essentially just the bf16
weights — leaving ~63 GB of an 80 GB H100 idle. Memory grows ~1.9 GB per unit of
batch, so the true ceiling is far above 8; the practical limit is the effective-
batch constraint below, not the card.

## Design

Two arms on the **identical rows Arm A trained on**
(`artifacts/twotower_rrf_triplet_ablation/rrf_003_multineg_k1.json`, mounted
read-only), both at effective batch 12 → the identical **245 optimizer steps**,
both at Qwen's own established `lr=1e-4` (not nano's 2e-4), 5 epochs, bf16 +
gradient checkpointing, H100:

| run id | micro-batch | accum | role |
|---|---|---|---|
| `qwen_micro1_r1` | 1 | 12 | control — the prior run's setting, at matched effective batch |
| `qwen_micro6_r1` | 6 | 2 | treatment — Arm A's exact micro-batch |

Only micro-batch differs, so this reads exactly like Arm C vs Arm A.

Because micro-batch must divide 12 to keep optimizer-step counts matched, 6 is
the largest useful setting below 12 — not a memory limit.

## An aborted first attempt, and what survived

The first launch (`qwen_micro6`, `qwen_micro1`) was killed by Modal at ~92%
(epoch 4.3 of 5) because the launcher process was backgrounded and its
termination cancelled the apps despite `--detach`. Operator error, not a code or
memory fault.

`save_total_limit=5` meant **all four completed epochs survived** on the volume,
along with their per-epoch dev evals — so the runs were substantially
salvageable, and resuming from `checkpoint-196` (~49 steps, ~6 min) would have
been far cheaper than the clean re-run actually performed (~30 min/arm). Noted
so the next interruption is handled better.

Dev accuracy recovered from that attempt (60-row synthetic dev set):

| arm | ep1 | ep2 | ep3 | ep4 |
|---|---|---|---|---|
| `qwen_micro1` | 0.5000 | 0.6833 | 0.6667 | 0.6833 |
| `qwen_micro6` | 0.4667 | 0.5833 | 0.6000 | 0.6333 |

Both learn, and both clear the nano arms' 0.50–0.58 ceiling — the 8B backbone
fits the synthetic task much better.

**Micro1's lead here should not be read as micro1 winning.** The nano ablation
produced the same pattern inverted from its own result: Arm C (micro-batch 2)
beat Arm A (micro-batch 6) on dev at every epoch — 0.550/0.567/0.567/0.567/0.583
vs 0.500/0.533/0.533/0.550/0.550 — and Arm A still won every real holdout
metric. The dev set is synthetic and drawn from the training distribution; a
model trained against more in-batch negatives solves a harder objective, fits
that distribution less tightly, and generalises better. In this project's one
prior test, higher dev accuracy predicted *worse* real performance.

## Results

### Real 69-pair holdout

| arm | pair AUC | hard-neg | MRR | R@1 | R@10 |
|---|---|---|---|---|---|
| micro-batch 1 (control) | 0.6853 | 0.6224 | 0.4985 | 0.3448 | 0.8621 |
| micro-batch 6 (treatment) | 0.6776 | **0.6379** | **0.5852** | **0.4483** | **0.8966** |
| **Δ (6 − 1)** | −0.0078 | +0.0155 | **+0.0867** | **+0.1034** | +0.0345 |

Both arms beat frozen Qwen and the prior micro-batch-1 fine-tune (0.6672) on
pair AUC, so training at effective batch 12 helps regardless of micro-batch.

### All 200 real pairs (corpus 178) — the honest test

| model | pair AUC | hard-neg | MRR | R@1 | R@10 |
|---|---|---|---|---|---|
| **qwen micro-6** | **0.5947** | 0.5608 | 0.3031 | 0.1400 | 0.6600 |
| voyage-4-large (prod) | 0.5726 | 0.5422 | 0.3102 | 0.1300 | **0.7000** |
| nano Arm A (v2) | 0.5594 | 0.5558 | **0.3341** | **0.1800** | 0.6400 |
| qwen micro-1 | 0.5604 | 0.4828 | 0.2734 | 0.1400 | 0.5600 |
| nano frozen | 0.5593 | 0.5046 | 0.3171 | 0.1800 | 0.5900 |
| qwen frozen 4096 | 0.5420 | 0.4572 | 0.2031 | 0.0400 | 0.5800 |
| qwen frozen 1024 | 0.5278 | 0.4496 | 0.1951 | 0.0400 | 0.5300 |

## Verdict

**1. The micro-batch finding replicates on a 22× larger backbone.** Same
signature as the nano ablation — retrieval moves, pair AUC does not:

| | Δ pair AUC | Δ MRR | Δ R@1 |
|---|---|---|---|
| nano, micro 2→6 (holdout) | +0.005 | +0.052 | +0.069 |
| Qwen, micro 1→6 (holdout) | −0.008 | +0.087 | +0.103 |

On the larger samples micro-6's advantage shows up in classification instead
(+0.034 pair AUC / +0.078 hard-neg on all-200; +0.043 / +0.108 on train-131).
Micro-6 ≥ micro-1 in 14 of 15 metric×subset cells.

**2. Unlike nano, fine-tuning Qwen produces gains that generalise.** Against its
own frozen 4096-dim baseline on all 200 pairs: pair AUC +0.053, hard-neg +0.104,
MRR +0.100, R@1 +0.100, R@10 +0.080 — every one far above the ±0.014 noise floor.
The nano equivalent was **+0.0001 / +0.051 / +0.017 / +0.0000**. Fine-tuning an
8B backbone on this synthetic data works; fine-tuning nano moved pair AUC and
recall@1 by exactly nothing.

**3. The 69-pair holdout flatters Qwen ~4× more than it flatters nano.**
Holdout-minus-train pair AUC gap:

| model | holdout | train | gap |
|---|---|---|---|
| nano frozen | 0.5793 | 0.5507 | +0.029 |
| nano Arm A (v2) | 0.5983 | 0.5392 | +0.059 |
| qwen frozen 4096 | 0.6345 | 0.5134 | +0.121 |
| qwen micro-1 | 0.6828 | 0.5056 | +0.177 |
| qwen micro-6 | 0.6810 | 0.5484 | +0.133 |

**Qwen3-8B's published "beats Voyage-4-large" headline (0.6595 vs 0.6086) rests
entirely on the population where that bias is largest.** On all 200 real pairs,
frozen Qwen at full 4096 dims scores 0.5420 — *below* frozen nano's 0.5593 and
Voyage-large's 0.5726.

**4. Best model depends on the metric, and no model wins outright.** qwen micro-6
leads pair AUC; nano Arm A leads recall@1 and MRR at 1/23 the size; Voyage-large leads recall@10. Under the <100 ms budget the nano family is
the only one of the three that is clearly deployable.

**5. voyage-4-nano beats voyage-4-large at top-1 retrieval on real data** —
0.1800 vs 0.1300 on all 200, holding on all three populations. The credit belongs
entirely to nano, not the fine-tune: frozen nano and Arm A v2 both score exactly
0.1800.

## The truncation asymmetry (found late, stated plainly)

`truncate_dim=1024` was inherited from the nano preset. Its effect is not
symmetric:

| model | native dim | used | effect |
|---|---|---|---|
| voyage-4-nano | 1024 | 1024 | **no truncation** |
| Qwen3-Embedding-8B | 4096 | 1024 | **cut to ¼** |
| voyage-4-large | 2048 (Matryoshka) | 1024 | requested at half width |

So every nano-vs-Qwen comparison here has nano at full width and Qwen
handicapped. The `qwen_frozen_4096` control bounds the cost at +0.014 (all-200)
and +0.025 (train-131) on pair AUC — real but small, and not enough to change
any conclusion above. **The fine-tuned arms were trained at 1024 and have never
been tested at native width**, which makes an untruncated Qwen fine-tune the
clearest next experiment.

A residual worth recording: frozen Qwen at 4096 scores 0.6345 on the holdout
versus the published 0.6595, most likely bf16 here versus the baseline's fp32
path. Small, unexplained, and a reason not to paste these numbers next to
`docs/baseline-results-holdout.md` without the note.

## Evaluation protocol

Scored with `eval_real_full/`, on **all 200 real pairs** plus the train-131 and
holdout-69 subsets. That is sound here for the same reason it was for Arm A:
these runs trained purely on synthetic `rrf_003` rows and saw zero real pairs
and zero real profiles. `eval_real_full/guard.py` verifies this from each run's
`run_meta.json` and refuses adapters without all-synthetic provenance.

Two protocol notes:

- **The frozen Qwen control is evaluated in bf16 too**, so the fine-tune is
  never compared against a different-precision baseline. This means the frozen
  number here may not exactly equal the published 0.6595, which came from the
  `baselines/hf_embedding` path; both are reported.
- **Prompts are not passed explicitly.** Qwen3-Embedding-8B registers
  `"Instruct: …\nQuery:"` / `""` in its own `config_sentence_transformers.json`,
  byte-identical to what training used, so `encode_query`/`encode_document`
  apply the right ones. Passing them again would create a second, divergent
  source of truth.

## Caveats

- `rrf_003`'s labels are an LLM judge's opinion on synthetic profiles, not real
  accept/decline outcomes; judge accuracy on the hard slice is 0.5942.
- One run per arm so far, so there is no replicate-based noise floor for this
  experiment. The nano ablation's floor (pair AUC ±0.013–0.016) is the closest
  available reference and is not strictly transferable to a different backbone.
- Per `docs/eval-real-full-experiment.md`, retrieval metrics are not comparable
  across the three subsets — each ranks against a different-sized candidate pool
  (178 / 120 / 65).

## Reproduce

```bash
modal run twotower_qwen_bigbatch/probe_batch_size.py --batch-sizes 1,2,4,6,8

modal run --detach twotower_qwen_bigbatch/modal_train.py --run-id qwen_micro6_r1 \
    --train-batch-size 6 --gradient-accumulation-steps 2
modal run --detach twotower_qwen_bigbatch/modal_train.py --run-id qwen_micro1_r1 \
    --train-batch-size 1 --gradient-accumulation-steps 12

modal volume get dorby-twotower-qwen-bigbatch-checkpoints <run_id> \
    ./artifacts/twotower_qwen_bigbatch/<run_id>
modal run eval_real_full/modal_eval.py --run-id real200_qwen \
    --configs qwen_frozen,qwen_micro1,qwen_micro6
```

Keep the launching shell alive (`wait`), or `--detach` will still be cancelled
when the client process is terminated.
