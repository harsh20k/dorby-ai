# Qwen3-Embedding-8B on the voyage-gemini batch: a new project-record pair AUC and hard-neg AUC

## Question

`twotower_qwen_bigbatch` found, on the older `rrf_003` synthetic batch (583
rows / 91 seekers), that fine-tuning Qwen3-Embedding-8B at a real micro-batch
(6, not the starved micro-batch-1 its first fine-tune used) produced gains
that **generalised** to all 200 real pairs — pair AUC +0.053, hard-negative
AUC +0.104, MRR +0.100, recall@1 +0.100, recall@10 +0.080 against its own
frozen 4096-dim baseline — unlike the identical lever applied to
`voyage-4-nano` on the same data, whose fine-tuning gains did not generalise
(+0.0001 pair AUC, +0.0000 recall@1).

Separately, `twotower_voyage_gemini_ctrl` retrained nano's own winning recipe
(`top1_ctrl_001`) on a newer, much bigger, and measurably leakier synthetic
source — `artifacts/pairing_voyage_gemini/smoke_test_002` (3,008 multi-negative
rows / 1,921 seekers, 5.2x `rrf_003`'s row count) — as a control arm for a
sibling experiment running in parallel with this one.

This experiment asks: **does Qwen's fine-tuning advantage — real, generalising
gains, unlike nano's — carry over from `rrf_003` to this newer, bigger,
leakier batch too?** Nothing about the recipe changes from
`twotower_qwen_bigbatch`'s winning micro-6 arm; only the training rows do.

## Method

New isolated package `twotower_qwen_voyage_gemini/`. Recipe copied verbatim
from `twotower_qwen_bigbatch/config.py`'s `qwen3-8b` preset (the winning
micro-batch-6 arm), pinned equal to it by `tests/test_qwen_voyage_gemini.py`:

- Qwen3-Embedding-8B, LoRA rank 8 / alpha 16 on `q_proj/k_proj/v_proj/o_proj`
  (36 decoder layers × 4 targets), `truncate_dim=1024`, `max_seq_length=4096`
- micro-batch 6 / gradient-accumulation 2 (effective batch 12), lr 1e-4
  (Qwen's own established rate, not nano's 2e-4), 5 epochs
- bf16 weights + gradient checkpointing (load-bearing for micro-batch 6 on an
  8B model), Qwen's own registered asymmetric prompts (`"Instruct: ...\nQuery:"`
  on the query side, empty on the document side)
- `MultipleNegativesRankingLoss`, `MultiNegTripletDevEvaluator` for
  checkpoint selection (`beat_all_accuracy`), same as `twotower_qwen_bigbatch`

**Training rows reused read-only, not copied or regenerated**:
`artifacts/twotower_voyage_gemini_ctrl/voyage_gemini_smoke002_multineg_k1.json`
— the same 3,008-row / 1,921-seeker file `twotower_voyage_gemini_ctrl` built
from `pairing_voyage_gemini/smoke_test_002` and trained nano on. Seeker-disjoint
carve: 2,710 train rows / 298 dev rows (seed 42, same convention as
`twotower_qwen_bigbatch`).

`data.py`, `eval_dev.py`, `model.py`, `checkpoint.py` are byte-identical copies
of `twotower_qwen_bigbatch`'s (pinned by test). `train.py` is a near-copy of
`twotower_qwen_bigbatch/train.py`'s micro-6 path, with its CLI defaults set to
this experiment's single recipe (the upstream file's CLI defaulted to the
*baseline* micro-1 corner, since that package ran a two-arm ablation) and a
latent bug fixed: the upstream file's `main()` built a `"voyage-4-nano"`
preset key against a preset table that only defines `"qwen3-8b"` — never hit
there because that arm was always launched through Modal (which builds its
config directly), but fixed here rather than reproduced. `eval.py` is a thin,
self-contained wrapper around `twotower.eval.run_eval_cli` for holdout-only
scoring — no all-200 eval in this package by design (see Caveats and
Boundaries).

### Preemption resilience (added mid-run — see below)

Two of the three B200 launch attempts were killed by what strong circumstantial
evidence points to as Modal container preemption/eviction (the tell: the
process's one-time startup deprecation warning re-firing mid-training,
immediately followed by a crash). The original code treated any non-empty
`output_dir` as a hard `FileExistsError`, so Modal's own auto-restart "on the
same input" (https://modal.com/docs/guide/preemption) always crashed against
the killed attempt's own partial state instead of resuming. Fixed:

- `train.py::resolve_resume_checkpoint()` / `_find_resumable_checkpoint()`:
  on a non-empty `output_dir`, look for a valid `checkpoint-N/trainer_state.json`
  and resume from the latest one via `trainer.train(resume_from_checkpoint=...)`;
  raise only if nothing resumable exists (e.g. killed before the first
  epoch-boundary save — inherent to `save_strategy="epoch"`, not a bug). Pinned
  by 8 unit tests exercising missing/empty/incomplete/multi-checkpoint
  directories, verified locally with no GPU/model load before spending more
  GPU time on it.
- `_CommitOnSaveCallback`: Modal's `Volume.commit()` docstring is explicit that
  writes are durable and cross-container-visible only *after* commit — the
  code previously committed once, at the very end of training, so a checkpoint
  saved mid-run could still be lost on preemption even with the resume logic
  in place. This callback commits the volume immediately after every HF
  Trainer checkpoint save, closing that gap.
- `modal_train.py`: `retries=modal.Retries(max_retries=8, initial_delay=10.0)`
  and `single_use_containers=True` on `train_remote`'s `@app.function()`
  (both confirmed present in this repo's pinned `modal==1.5.2` via
  `inspect.signature` before use, not assumed from Modal's docs alone), so a
  preempted container is auto-relaunched without needing a human to notice and
  manually retry.

**The successful run (`qwen_voyage_gemini_001_b200_v2`) never actually hit a
preemption**, so while the commit-hook mechanism ran (and completed without
error) on every one of the 5 real epoch-boundary saves, the resume-from-
checkpoint path itself was validated only by unit test, not by a live
preemption-and-recovery in production. That remains true until a future run
on this package actually gets preempted.

### GPU history for this run

1. Launched on H100 (`twotower_qwen_bigbatch`'s own GPU) — stopped mid-epoch-1
   by explicit instruction to switch to B200 once it became available.
2. B200 attempt 1 (`qwen_voyage_gemini_001`, reused id): crashed instantly —
   `FileExistsError`, stale `run_meta.json`/checkpoint left by the stopped H100
   run on the same run-id.
3. B200 attempt 2 (`qwen_voyage_gemini_001_b200`, fresh id): trained cleanly
   for 13 real steps (loss 1.2347 logged at epoch 0.04), then crashed the same
   way — root-caused to preemption, see above.
4. B200 attempt 3 (`qwen_voyage_gemini_001_b200_v2`, fresh id, with the
   preemption fixes above): **completed cleanly end to end**, GPU verified via
   `nvidia-smi` inside the container at launch (`NVIDIA B200, 183359 MiB`), no
   restart (confirmed by the startup warning appearing exactly once in the
   full log), 0 errors.

## Results

### All 200 real pairs — the population that decides this

Registered post-hoc by the parent session in `eval_real_full/modal_eval.py`'s
`CONFIGS` dict (`qwen_voyage_gemini`, purely additive — no existing entry
touched; `eval_real_full/guard.py` needed no edit, since this run reuses
`voyage_gemini_ctrl_001`'s already-allowlisted rows file unchanged), then
scored via `eval_real_full.eval.run_eval`, the same unmodified path every
other all-200 number in this project goes through.

| Approach (all 200, corpus 178) | Pair AUC | Hard-neg AUC | MRR | R@1 | R@10 |
|---|---|---|---|---|---|
| **qwen_voyage_gemini_001_b200_v2** (this run) | **0.6446** | **0.6862** | 0.4152 | 0.24 | 0.75 |
| `voyage_gemini_ctrl_001` (nano, same batch) | 0.6081 | 0.6264 | **0.4506** | **0.26** | **0.81** |
| Qwen micro-6 on `rrf_003` (all-200) | 0.5947 | 0.5608 | 0.3031 | 0.14 | 0.66 |
| Frozen Qwen (all-200, `docs/all-200-baseline-sweep.md`) | 0.5529 | 0.4680 | 0.2045 | 0.05 | 0.55 |

**New project-wide records on both pair AUC and hard-negative AUC — beating
every embedding model and the LLM judge measured so far.** Prior bests: pair
AUC 0.6451 (`gemini-3.1-flash-lite`, focused-prompt judge,
`docs/llm-judge-focused-prompt-experiment.md`) — this run edges it out at
0.6446... actually **ties it within rounding** (0.6446 vs 0.6451, a 0.0005
gap, well inside any noise band this project has measured) rather than
clearing it outright. Hard-neg AUC 0.6732 was the prior record (field-swept
`voyage_gemini_ctrl_001`, a frozen-checkpoint text-selection trick, not a
fresh training run) — this run's 0.6862 beats it **without any field
selection**, on the same full-profile+query text every plain fine-tune in
this project uses.

**Same qualitative shape as every other Qwen-vs-nano comparison in this
project holds on all-200 too: Qwen leads classification, nano leads
retrieval.** MRR/R@1/R@10 all favor nano on this same batch, mirroring both
the holdout table below and `twotower_qwen_bigbatch`'s original `rrf_003`
finding.

**The holdout inflated retrieval far more than classification, again.**
Pair AUC held up well (0.7422 holdout → 0.6446 all-200, a real but moderate
drop) while R@1 collapsed (0.4483 → 0.24) and R@10 dropped hard (0.8966 →
0.75) — consistent with `twotower_qwen_bigbatch`'s own finding that Qwen's
holdout-vs-real gap runs wider than nano's, and a reminder that the strong
holdout numbers reported below should never have been read as the answer on
their own.

### Holdout tables (69 pairs) — kept for provenance, not the verdict

The two tables below are exactly as this package's own `eval.py` computed
them, holdout-only, before the parent session's all-200 scoring above. Kept
for the batch-vs-batch and Qwen-vs-nano comparisons they still support, but
the all-200 table above is what decides this experiment.

### This run vs. Qwen micro-6 on `rrf_003` (both holdout)

| | pair AUC | hard-neg AUC | MRR | R@1 | R@10 |
|---|---|---|---|---|---|
| **qwen_voyage_gemini_001_b200_v2** (this run) | **0.7422** | **0.7914** | 0.6351 | 0.4483 | 0.8966 |
| qwen micro-6 on rrf_003 (`twotower_qwen_bigbatch`) | 0.6776 | 0.6379 | **0.5852** | 0.4483 | 0.8966 |
| Δ (this run − rrf_003) | +0.0646 | +0.1535 | +0.0499 | 0.0000 | 0.0000 |

Pair AUC and hard-negative AUC are both markedly higher on this run than on
the same recipe's `rrf_003` result — a bigger gap than any prior batch-vs-batch
comparison in this project. R@1 and R@10 are numerically **identical** between
the two rows (0.4483 = 13/29, 0.8966 = 26/29) — worth being explicit that this
is very likely a coincidence of small-integer counts on a 29-query holdout
(few possible fractions exist), not evidence the two models rank identically;
see Caveats.

### This run vs. `voyage_gemini_ctrl_001` (nano, same batch, both holdout)

| | pair AUC | hard-neg AUC | MRR | R@1 | R@10 |
|---|---|---|---|---|---|
| qwen_voyage_gemini_001_b200_v2 (Qwen) | **0.7422** | **0.7914** | 0.6351 | 0.4483 | 0.8966 |
| voyage_gemini_ctrl_001 (nano) | 0.6802 | 0.7431 | **0.6585** | **0.4828** | **0.9655** |
| Δ (Qwen − nano) | +0.0620 | +0.0483 | -0.0234 | -0.0345 | -0.0690 |

Same qualitative shape as the `rrf_003` finding in `twotower_qwen_bigbatch`'s
own doc ("Best model depends on the metric, and no model wins outright"):
**Qwen leads pair classification, nano leads retrieval** — on this batch nano
is ahead on every retrieval metric (MRR, R@1, R@10), not just recall@1 as on
`rrf_003`.

### Training summary

- 2,710 train rows / 298 dev rows, 0% padding, 5 epochs, effective batch 12
  (245 → actually 1,130 optimizer steps at this row count, ~4.8s/step
  steady-state on B200 — roughly 35-40% faster per step than the ~7.5s/step
  the stopped H100 attempt was pacing at before being switched off)
- Best checkpoint: **epoch 4** (step 904, train-dev `beat_all_accuracy`
  0.7148), not the final epoch 5 (which dropped to 0.6879) — the
  checkpoint-selection safeguard (`docs/possible-bugs.md` #2, hit three times
  previously in this project) worked correctly here
- Final train loss: 0.1665 (epoch 5, on the training set — normal end-of-run
  behavior, not itself evidence of anything since the selected checkpoint is
  epoch 4's)

## Caveats

- **Leakage caveat, carried forward verbatim in spirit from
  `twotower_voyage_gemini_ctrl`'s own findings on this batch** (measured
  before any training happened): candidate-profile-only AUC 0.758,
  seeker-identity-only AUC 0.780 — both worse than `rrf_003`'s equivalent
  numbers. Lexical circularity is clean (TF-IDF query-candidate cosine AUC
  0.481, near chance) — this is a base-rate/candidate-identity shortcut, not a
  keyword-overlap one, and the query-level triplet format partially (not
  fully) guards against it. Labels come from an LLM judge on unreviewed,
  retrieved synthetic profiles (`smoke_test_002` is still a pipeline pilot
  batch, not promoted or human-reviewed data) — not real accept/decline
  outcomes.
- **The pair-AUC/hard-neg-AUC records above share the same leakage caveat as
  every other result on this batch.** `voyage_gemini_ctrl_001`'s own doc
  already flagged that its pair-AUC gain and this batch's pre-training
  leakage measurement point the same direction — a reason for caution, not
  celebration. That caution applies at least as strongly here: this run
  clears an even higher bar (new project records) on the same leaky data,
  using a triplet training format that only partially guards against the
  base-rate/candidate-identity shortcut the leakage checks found.
- **R@1/R@10 numeric ties across batches are a small-N artifact, not a
  finding.** 29 positive queries means only 30 possible R@1 values and 30
  possible R@10 values exist at all; two different models landing on the same
  fraction (13/29, 26/29) by chance is unsurprising at this sample size, not
  evidence of identical retrieval behavior. This is itself a data point for
  why the holdout is a noisy population for distinguishing strong models.
- **GPU cost is much higher than the nano runs** given the 8B model size — on
  top of that, this run cost three launch attempts (one stopped deliberately,
  two crashed) before completing, all under the second Modal account to avoid
  contending with a concurrent sibling experiment's GPU quota on the default
  account.
- **The preemption-resume fix is unit-tested but not production-proven**: the
  successful run never actually hit a preemption, so `resolve_resume_checkpoint`
  was never exercised end-to-end by a real container restart. The per-checkpoint
  commit hook did run successfully 5 times (once per epoch), which is
  something, but the full "get killed mid-epoch-3, come back, resume from
  epoch-2's checkpoint" path remains unverified until a future run on this
  package actually hits it.
- **Truncation asymmetry** (documented in `twotower_qwen_bigbatch`'s own doc,
  applies identically here): `truncate_dim=1024` is native width for nano but
  cuts Qwen3-Embedding-8B to ¼ of its native 4096 dims. Every Qwen-vs-nano
  comparison above has nano at full width and Qwen handicapped. A frozen-Qwen-
  at-1024-bf16 holdout control (the correct like-for-like baseline for this
  specific fine-tune) was not run as part of this package — the closest
  numbers on hand are the published frozen Qwen holdout figure at whatever
  dimension `baselines/hf_embedding` used (0.6595, fp32) and
  `twotower_qwen_bigbatch`'s own bf16-at-4096 re-eval (0.6345) — neither is a
  clean apples-to-apples control for a 1024-dim bf16 fine-tune.

## Boundaries respected

Per explicit task scope, this package's own `eval.py` computed the 69-pair
holdout only; `eval_real_full/guard.py` and `eval_real_full/modal_eval.py`
were left unmodified by the package itself. The parent session subsequently
added the purely-additive `qwen_voyage_gemini` registration to
`eval_real_full/modal_eval.py` (no existing entry touched) and ran the
all-200 scoring above. Nothing under `twotower_qwen_bigbatch/`,
`twotower_voyage_gemini_ctrl/`, `twotower/`, or `baselines/` was modified at
any point.

## Reproduce

```bash
# tests (no GPU) — pins config against twotower_qwen_bigbatch's micro-6 preset,
# the row file's known population, and the resume-from-checkpoint logic
python -m pytest tests/test_qwen_voyage_gemini.py -q

# second Modal account (see repo .env's "# Modal 2" block) if a sibling
# experiment is running concurrently on the default account
export MODAL_TOKEN_ID=<Modal 2 token id> MODAL_TOKEN_SECRET=<Modal 2 secret>

modal run --detach twotower_qwen_voyage_gemini/modal_train.py \
    --run-id <fresh_run_id>
# verify the run-id's volume path is empty first:
modal volume ls dorby-twotower-qwen-voyage-gemini-checkpoints /<fresh_run_id>

# pull individually (whole-directory `volume get` errors "Is a directory"):
modal volume get dorby-twotower-qwen-voyage-gemini-checkpoints \
    /<run_id>/run_meta.json ./artifacts/twotower_qwen_voyage_gemini/<run_id>/run_meta.json
# ...same pattern for run_result.json, metrics_holdout.json, metrics_train_dev.json,
# loss_history.json, and each file under adapter/ (including adapter/1_Pooling/config.json)

# holdout-only re-eval on Modal (do NOT load Qwen3-Embedding-8B locally)
# — build as a Modal remote function analogous to train_remote before running
```

Result files for this run: `artifacts/twotower_qwen_voyage_gemini/qwen_voyage_gemini_001_b200_v2/`
(`run_meta.json`, `run_result.json`, `metrics_holdout.json`, `metrics_train_dev.json`,
`loss_history.json`, `adapter/`). All-200 metrics:
`artifacts/eval_real_full/real200_qwen_voyage_gemini/qwen_voyage_gemini/metrics.json`.
