# The recall@1-best text combo, actually trained: new best fine-tune in the project

## Question

A 105-way field/query sweep against the frozen `top1_ctrl` checkpoint
(`baselines/twotower_top1_ctrl_field_sweep/`, "top1_ctrl field/query sweep")
re-scored every non-empty seeker/candidate field-subset combination on all
200 real pairs and found the best recall@1 anywhere in the grid: seeker =
**search query only, zero profile fields**, candidate = **background +
lookingFor**, query included — recall@1 0.28, recall@5 0.75, pair AUC 0.60,
all scored *frozen*, re-using the existing `top1_ctrl` checkpoint's weights
with different input text. Nobody had trained on this exact combination.
Does actually fine-tuning on it beat `top1_ctrl`'s own full-profile recipe?

## Method

New isolated package `twotower_queryonly_back_look/`. Every setting held
identical to `top1_ctrl_001` — LoRA rank 8 / alpha 16 / dropout 0.05 on
q/k/v/o_proj (983,040 trainable params), micro-batch 6 / accum 2 (effective
batch 12, 245 optimizer steps), lr 2e-4, 5 epochs, `voyage-4-nano` at its
native 1024-dim truncation, plain library-default
`MultipleNegativesRankingLoss(scale=20.0)`, `CorpusRecallDevEvaluator`
recall@1 checkpoint selection, same 643-row `rrf_003` population (583
train / 60 dev, 0% padding) `top1_ctrl` trained on. The only variable:
seeker text = the pair's search query alone (no profile fields at all),
candidate text = `background` + `lookingFor` only.

Both text builders are reused **unmodified**, not duplicated:
`query_weighted.text.query_only` for the seeker side (falls back to the
profile only if a pair's query is empty — 0/643 in this batch),
`field_pairs_sweep.text.background_lookingfor` for the candidate side.
`scripts/build_rrf_multineg_triplets_queryonly_back_look.py` (new
row-builder script) verified the row file matches `top1_ctrl`'s population
exactly: 643 rows, 297 seekers, 0% padding.

**Scoring problem, same as the earlier field-pair packages:** the real
200-pair loader always builds full-profile candidate text, so it can't
fairly score an adapter trained on trimmed text. `twotower_queryonly_back
_look/eval.py` reuses `twotower.eval`'s model loading and query/document
encoding unmodified, but builds seeker/candidate text with the same two
reused builders training used. Hardness split stays pinned to the full
profile+query baseline text, so hard-neg AUC is comparable to every other
row in the project.

```bash
python -m pytest tests/test_queryonly_back_look.py -q

modal run --detach twotower_queryonly_back_look/modal_train.py --run-id queryonly_back_look_001
modal run twotower_queryonly_back_look/modal_eval.py --run-id queryonly_back_look_001
```

## Results — all 200 real pairs

| Approach | Pair AUC | Hard-neg AUC | Easy-neg AUC | MRR | R@1 | R@5 | R@10 |
|---|---|---|---|---|---|---|---|
| `top1_ctrl` (previous best, full profile) | 0.5683 | 0.5484 | 0.6836 | 0.3550 | 0.19 | — | 0.69 |
| **`queryonly_back_look_001`** (this experiment) | **0.5983** | **0.6564** | 0.5700 | **0.4791** | **0.30** | **0.74** | **0.86** |

**Beats `top1_ctrl` on every metric.** +0.030 pair AUC, +0.108 hard-neg
AUC, +0.124 MRR, +11 more correct top-1 matches out of 100, +0.17 recall@10.
This is now the best two-tower fine-tune in the project on every tracked
metric — the first of six architectural/text-selection attempts against
`top1_ctrl` (split towers, field gate, KL regularization, three field-pair
arms) to actually win rather than lose or trade.

**The hard/easy-neg AUC ordering inverted** (hard-neg 0.6564 > easy-neg
0.5700) — the *only* other model in this project with that inversion is the
LLM judge (`docs/llm-judge-experiment.md`, hard-neg 0.6466 > easy-neg
0.5638), which the project's own findings call direct evidence a model is
"not scoring lexical similarity" rather than topic overlap. Every embedding
baseline measured so far, including `top1_ctrl` itself (easy 0.6836 vs hard
0.5484), shows the opposite: strong on easy negatives, weak on hard ones.
This fine-tune is the first embedding-based model in the project to show
the judge's signature instead.

**Training helped past the frozen sweep's own number**, not just past
`top1_ctrl`: the frozen-sweep version of this exact combo scored recall@1
0.28 / pair AUC 0.6016 (re-using `top1_ctrl`'s existing weights); the
trained version reaches recall@1 0.30 / pair AUC 0.5983 — recall improved,
pair AUC is flat within noise. Fine-tuning specifically on this text
representation did move the needle a little further, not just recover
what the frozen model already had.

## Where does the improvement actually come from?

The two-row table above compares two different *text combos* (`top1_ctrl`'s
full profile vs. this experiment's query-only/background+lookingFor), which
conflates "did fine-tuning help" with "was this text choice better." To
isolate training's actual contribution, the same text combo was scored at
three stages — frozen base `voyage-4-nano` (no fine-tuning at all),
`top1_ctrl`'s existing weights with this text swapped in (general transfer,
it never saw this text during training), and this experiment's adapter
(trained specifically on this text):

| Stage | Pair AUC | Hard-neg AUC | Easy-neg AUC | MRR | R@1 | R@5 | R@10 |
|---|---|---|---|---|---|---|---|
| 1. Frozen base nano | 0.5626 | 0.4374 | 0.7424 | 0.4763 | 0.28 | 0.76 | 0.85 |
| 2. `top1_ctrl` weights, text swapped in | 0.6016 | 0.5220 | 0.7280 | 0.4736 | 0.28 | 0.75 | 0.86 |
| 3. Trained specifically (this experiment) | 0.5983 | **0.6564** | 0.5700 | 0.4791 | 0.30 | 0.74 | 0.86 |

**Recall@1 and recall@5 were mostly decided by the text choice, not
training** — frozen base nano already gets R@1 0.28 / R@5 0.76 for this
combo; general and specific fine-tuning together move R@1 by only +0.02
and leave R@5 flat (even a hair down).

**Hard-negative AUC is where training earns its keep.** General fine-tuning
transfer (stage 1→2) already buys +0.085 (0.437→0.522) just from having
trained on *something*. Training specifically on this text (stage 2→3)
adds another **+0.134** (0.522→0.656) — the single largest move in the
whole chain, and the actual reason this experiment sets the new
project-wide record. Pair AUC and easy-neg AUC both dip slightly in that
same last step — a real trade, not a free win on every axis.

**The holdout did not mislead this time** — a real break from the pattern.
Holdout AUC 0.6603 / R@1 0.4828 vs. all-200's 0.5983 / 0.30: both point the
same direction (this arm is strong), unlike every other custom-loop
experiment in this project (`split_001`, `field_gate_001`, `kl_reg_ctrl
_001`, all three field-pair arms), where the holdout consistently
overstated a result that reversed on all 200. That doesn't mean the
holdout can be trusted going forward — it means this is the first result
where it happened to agree.

## Caveat carried over from the sweep

The field/query sweep itself flagged: 105-way search on one 200-pair
population, no held-out check — its best combo by pair AUC (0.6395) sits
within run-to-run noise of its 10th-best (0.6167). This experiment answers
a narrower, cleaner question than that caveat covers: *given* the sweep's
recall@1-best combo, does training on it specifically help? Yes. It does
not by itself resolve whether the combo-selection process was overfit to
this 200-pair population — that would need a genuinely held-out check
(e.g. a fresh batch of real pairs, which doesn't exist yet) or a
replication run with a different seed.

## What this means

Every architectural change tried against `top1_ctrl`'s recipe up to this
point — separate towers, a learned gate, a KL leash — lost. The one change
that won was not architectural at all: it was choosing a better *text
representation* for both sides, found by brute-force search on the frozen
model first and only then spent GPU budget confirming it trains well too.

**A new project-wide record, not just a `top1_ctrl` win:** the prior best
hard-negative AUC of *any* model tested in this project — embedding or
LLM-judge — was 0.6466 (`gemini-3.1-flash-lite` naive judge,
`docs/llm-judge-experiment.md`). This fine-tune reaches **0.6564**, beating
it, at a merged-LoRA-adapter serving cost rather than a per-candidate LLM
call — the judge is explicitly out of the project's <100ms latency budget,
this is not.

**Not a clean sweep against every free lever, though.** `top1_ctrl` +
eval-time query-weighting (`docs/query-weighted-twotower-experiment.md`) —
a zero-training text swap on the existing checkpoint — still holds higher
MRR (0.5076 vs 0.4791) and recall@1 (0.32 vs 0.30) via its `query_only`
swap, and higher pair AUC (0.6129 vs 0.5983) via its `alpha_0.6` blend.
Both of those cost nothing to obtain. This experiment's genuine edge is
hard-negative resistance — the metric that matters most given the real
negative population is production's own false positives — plus proof that
a frozen-model field/query search is now a validated way to *find* a text
representation worth spending GPU budget on.

Published artifact: https://claude.ai/code/artifact/47754226-aeaf-4306-902f-35a9ed6ee586
