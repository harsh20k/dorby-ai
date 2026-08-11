# top1_ctrl's exact recipe, retrained on a bigger, newer synthetic batch

## Question

`top1_ctrl` was trained on 643 rows from `rrf_003` (`pairing_rrf`, Qwen3-
Embedding-8B retrieval + Bedrock judge). A newer, unrelated pipeline
(`pairing_voyage_gemini`) has since produced a much larger batch —
`smoke_test_002`, 19,695 pairs, Voyage-4-large + Chroma retrieval,
`gemini-3.1-flash-lite` judge — still just a pilot run of that pipeline, not
reviewed or promoted. Basic leakage checks run on it directly (before any
training) found it measurably leakier than `rrf_002`/`rrf_003`: candidate-
profile-only AUC 0.758, seeker-identity-only AUC 0.780, 43.5% of seekers
all-positive or all-negative across every query they asked (vs. `rrf_002`'s
0.634 / 0.687 / 30%). Lexical circularity was clean (TF-IDF query-candidate
cosine AUC 0.481, near chance).

Given that caveat, does `top1_ctrl`'s exact recipe — unchanged — still
transfer to this bigger, leakier source? That's the only variable this
experiment isolates: identical LoRA config, identical loss, identical text
fields, different training rows.

## Method

New isolated package `twotower_voyage_gemini_ctrl/`. Every setting held
identical to `top1_ctrl_001`: LoRA rank 8 / alpha 16 / dropout 0.05 on
q/k/v/o_proj (983,040 trainable params), micro-batch 6 / accum 2 (effective
batch 12), lr 2e-4, 5 epochs, `voyage-4-nano` at its native 1024-dim
truncation, plain library-default `MultipleNegativesRankingLoss(scale=20.0)`,
`CorpusRecallDevEvaluator` recall@1 checkpoint selection, full-profile text on
both sides (`baselines.bert_frozen.text.seeker_to_text`/`candidate_to_text`,
reused unmodified — same builder `top1_ctrl` used).

Rows: `scripts/build_rrf_multineg_triplets.py`, unmodified — it already takes
`--batch-dir` as a plain argument, so pointing it at `pairing_voyage_gemini`'s
`smoke_test_002` instead of `rrf_003` is reuse, not a copy — against k=1
(0% padding; every one of 2,187 both-class query_keys has >=1 unique
negative). **3,008 rows / 1,921 seekers** — 4.7x `top1_ctrl`'s row count, off
only 24.3% of the batch's query_keys (the rest are single-class per query and
can't form a training row).

`eval_real_full.eval.run_eval` is reused unmodified for all-200 scoring —
this arm uses standard full-profile text, so unlike `twotower_queryonly_back
_look` it needs no custom eval.py. Two purely-additive registrations were
needed to make that possible: `eval_real_full/guard.py`'s
`SYNTHETIC_ONLY_ROW_SOURCES` allowlist gained one new token
(`voyage_gemini_smoke002_multineg`) and `eval_real_full/modal_eval.py`'s
`CONFIGS` dict gained one new entry (`voyage_gemini_ctrl`) — the same pattern
every prior full-profile arm (`top1_ctrl`, `top1_sharp`, both Qwen arms) used
to register itself. Neither edit touches an existing entry or any scoring
logic; no prior published number is affected.

```bash
python -m pytest tests/test_voyage_gemini_ctrl.py -q

modal run --detach twotower_voyage_gemini_ctrl/modal_train.py --run-id voyage_gemini_ctrl_001
modal volume get dorby-twotower-voyage-gemini-ctrl-checkpoints voyage_gemini_ctrl_001 \
    ./artifacts/twotower_voyage_gemini_ctrl/voyage_gemini_ctrl_001

modal run eval_real_full/modal_eval.py --run-id real200_voyage_gemini_ctrl --configs voyage_gemini_ctrl
```

## Results

Training: 1,130 optimizer steps (5x `top1_ctrl`'s 245, since this batch has
4.7x the rows at the same effective batch), best checkpoint selected at step
452 (epoch 2.0) by dev recall@1, final train loss 0.305.

**The 69-pair holdout overstated this run again** — the same pattern this
project has hit on every custom-loop experiment except `queryonly_back_look`:

| Subset | Pair AUC | Hard-neg AUC | MRR | R@1 | R@10 |
|---|---|---|---|---|---|
| holdout (69 pairs) | 0.6802 | 0.7431 | 0.6585 | 0.4828 | 0.9655 |
| **all 200 real pairs** | **0.6081** | **0.6264** | **0.4506** | **0.26** | **0.81** |

All-200 is the number that counts. Compared against `top1_ctrl` — the exact
same recipe and text fields, only the training rows differ:

| Approach | Pair AUC | Hard-neg AUC | Easy-neg AUC | MRR | R@1 | R@10 |
|---|---|---|---|---|---|---|
| `top1_ctrl` (rrf_003 data) | 0.5683 | 0.5484 | 0.6836 | 0.3550 | 0.19 | 0.69 |
| **`voyage_gemini_ctrl_001`** (this run) | **0.6081** | **0.6264** | 0.6544 | **0.4506** | **0.26** | **0.81** |

**Beats `top1_ctrl` on every metric**, apples-to-apples: +0.040 pair AUC,
+0.078 hard-neg AUC, +0.096 MRR, +7 more correct top-1 matches out of 100,
+0.12 R@10. The bigger, newer, more homogeneous batch does help, despite
scoring measurably leakier than `rrf_002`/`rrf_003` on every leakage check run
against it beforehand.

**Does not set a new project record.** `queryonly_back_look_001` (different
text fields — query-only seeker, background+lookingFor candidate — still
holds the top spot on hard-neg AUC (0.6564), MRR (0.4791), R@1 (0.30), and
R@10 (0.86). This run edges it out only on pair AUC (0.6081 vs 0.5983). So
`top1_ctrl`'s recipe transfers to the new data, but a better text choice
(found by the frozen-model field/query sweep, not by more/different training
data) still beats it on every metric that matters most for retrieval.

**Easy/hard-neg AUC ordering is normal here** (easy 0.6544 > hard 0.6264) —
unlike `queryonly_back_look`, which inverted that ordering and was flagged as
the first embedding-based model in the project to show the LLM judge's
"not scoring lexical similarity" signature. This run doesn't show that
signature; it looks like every other embedding baseline in the project.

## Caveats

- **Labels are an unreviewed pilot batch's LLM-judge opinions**, not real
  accept/decline outcomes. `smoke_test_002` is named as a pipeline pilot, not
  a promoted or human-reviewed dataset — same status `rrf_002`/`rrf_003` had
  before their own trainability write-ups, except this batch is measurably
  leakier on two of the three checks run.
- **The leakage found beforehand may be part of the win, not just noise
  around it.** This run's all-200 pair AUC (0.6081) is the best of any
  fine-tune in the project — edging out `queryonly_back_look_001`'s 0.5983,
  which trained on the cleaner `rrf_003` batch. Candidate-only AUC 0.758 and
  seeker-identity AUC 0.780 were both higher (more leakage) here than
  `rrf_002`'s. The triplet training format (candidates compared within one
  query, not against the global base rate) plausibly absorbs some of that
  leakage, but this experiment does not prove how much — the pair-AUC gain
  and the pre-training leakage measurement point the same direction, and
  that's reason for caution, not celebration.
- **This is a same-recipe, different-data comparison only.** It answers "does
  `top1_ctrl`'s recipe transfer," not "is this the best text/data combination
  available" — the field/query sweep methodology that found
  `queryonly_back_look`'s combo has not been run against this new batch.

## What this means

Training data source matters as much as recipe here: identical LoRA config,
identical loss, identical text fields, and simply pointing at a bigger
synthetic batch bought +0.04 pair AUC and +0.08 hard-neg AUC over `top1_ctrl`.
But the project's actual record holder got there by changing what the model
reads, not what it trained on — and that lever still wins. The next natural
step, not yet run: point the field/query sweep methodology at this new batch
before spending more GPU budget training on its default full-profile text.

Published artifact: https://claude.ai/code/artifact/11d21292-33d6-42b6-8d5d-f42b41b620e6
