"""top1_ctrl's exact recipe, retrained on the pairing_voyage_gemini batch.

Same question `twotower_top1_optimised` answered for `rrf_003`, asked again for
a newer, much larger synthetic source: `artifacts/pairing_voyage_gemini/
smoke_test_002` (19,695 pairs, retrieved via Voyage-4-large + Chroma, labeled by
the `gemini-3.1-flash-lite` judge — a different pipeline from `pairing_rrf`,
still a same-shape LLM-judge-on-retrieved-candidates design).

Every setting held identical to `top1_ctrl_001`: LoRA rank 8 / alpha 16 /
dropout 0.05 on q/k/v/o_proj, micro-batch 6 / accum 2 (effective batch 12), lr
2e-4, 5 epochs, `voyage-4-nano` at its native 1024-dim truncation, plain
library-default `MultipleNegativesRankingLoss(scale=20.0, hardness_mode=None)`,
`CorpusRecallDevEvaluator` recall@1 checkpoint selection, full-profile text on
both sides (`baselines.bert_frozen.text.seeker_to_text`/`candidate_to_text`,
reused unmodified, same as `top1_ctrl` and every field-pair arm). The only
variable is the training rows.

Rows: `scripts/build_rrf_multineg_triplets.py` (unmodified — it already takes
`--batch-dir` as a plain argument, so pointing it at a different
`pairing_rrf`-shaped batch is reuse, not a copy) against
`artifacts/pairing_voyage_gemini/smoke_test_002`, k=1 negative/anchor (0%
padding — every one of 2,187 both-class query_keys has >=1 unique negative).
3,008 rows / 1,921 seekers — 4.7x `top1_ctrl`'s row count, 6.5x its seeker
count, off only 24.3% of this batch's query_keys (the rest are single-class per
query and can't form a training row).

**Leakage caveat found before this arm was built** (basic checks run against
`smoke_test_002` directly): candidate-profile-only AUC 0.758, seeker-identity-
only AUC 0.780, 43.5% of seekers are all-positive or all-negative across every
query they asked — all three worse than `rrf_002`, the batch `top1_ctrl`'s
population traces back to. Lexical circularity is clean (TF-IDF query-candidate
cosine AUC 0.481, near chance) — this batch is not a keyword-overlap shortcut,
but it is a bigger base-rate/candidate-identity shortcut than anything trained
on before. The query-level triplet format partially guards against this
(candidates are compared against each other within one query, not against the
global base rate) but does not eliminate it. Labels are also unreviewed — this
batch is still named `smoke_test_002`, a pipeline pilot, not a promoted or
human-reviewed dataset.

This experiment tests whether `top1_ctrl`'s recipe transfers to a different,
bigger, and measurably leakier synthetic source — not whether this batch is fit
to promote. Scoring is on all 200 real pairs only, via `eval_real_full.eval
.run_eval` reused unmodified (this arm uses standard full-profile text, so
unlike `twotower_queryonly_back_look` it does not need its own eval.py).
`eval_real_full/guard.py`'s `SYNTHETIC_ONLY_ROW_SOURCES` allowlist and
`eval_real_full/modal_eval.py`'s `CONFIGS` dict both gained one new entry for
this run — both are purely additive registration points already grown by every
prior full-profile arm (`top1_ctrl`, `top1_sharp`, the Qwen arms), not scoring
or training logic, so no prior published number is affected.
"""
