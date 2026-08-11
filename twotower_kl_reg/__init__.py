"""Isolated experiment: does a KL-to-frozen-base penalty fix the early-overfit
pattern seen in every custom-loop two-tower run so far (`run_001`, `split_001`,
`field_gate_001` — dev recall@1 peaks at epoch 1-2, then degrades every epoch
after despite training loss still falling)?

Everything is held identical to the project's best two-tower fine-tune,
`top1_ctrl` (`twotower_top1_optimised/`, run-id `top1_ctrl_001`): same
`rrf_003_multineg_k1.json` rows, one shared tower (seeker text = profile +
search query concatenated, candidate text = candidate profile only — the
standard `top1_ctrl` inputs, not the query-only/no-query/split variants tried
elsewhere), same LoRA shape, same batch/lr/epoch schedule, same
`MultipleNegativesRankingLoss(scale=20.0)` main term, same recall@1-based
checkpoint selection. The only change is one new loss term
(`losses.py::KLRegularizedMNRL`) that penalizes the LoRA-adapted model's
in-batch similarity distribution for drifting from the frozen base model's
same distribution — obtained by toggling the adapter off on the same PEFT
model (`disable_adapters()`/`enable_adapters()`), not a second model instance.

Isolated package: `data.py`/`eval_dev.py` are copies of
`twotower_top1_optimised/data.py`/`eval_dev.py` (pinned by
`tests/test_kl_reg.py` against drift), not imports — nothing under
`twotower_top1_optimised/`, `twotower/`, or any other prior experiment's
package is modified. `losses.py` is new. See
`docs/twotower-kl-reg-experiment.md` for the full writeup.
"""
