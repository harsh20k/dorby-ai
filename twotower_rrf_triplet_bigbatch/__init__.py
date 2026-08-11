"""Isolated re-run of the rrf_003 triplet fine-tune: bigger true batch size +
multiple negatives per anchor, for voyage-4-nano only.

Fully separate from both twotower/ (Arm A/B/C) and twotower_rrf_triplet/
(the first triplet fine-tune, rrf_triplet_voyage_nano_001 /
rrf_triplet_qwen3_8b_h100_002) — neither of those packages is imported here,
and this package writes only under artifacts/twotower_rrf_triplet_bigbatch/.
Only twotower/'s generic, unmodified helpers (build_model,
select_best_checkpoint, add_lora_adapter, smoke_backward, evaluate_pairs,
build_split_bundle, assert_no_holdout_leak) are reused, read-only, the same
way twotower_rrf_triplet/ itself reused them.

Rationale: rrf_triplet_voyage_nano_001 beat frozen Voyage-4-large on pair AUC
but trailed badly on retrieval (recall@1 0.276 vs 0.345). Its real per-step
batch size was only 2 (gradient accumulation gave an effective batch of 8,
but accumulation does not add in-batch negatives for
MultipleNegativesRankingLoss, only true batch size does). This experiment
tests whether a much larger true batch size plus multiple negatives per
anchor per row recovers some of that gap. See
docs/twotower-rrf-triplet-bigbatch-experiment.md for the full writeup.
"""
