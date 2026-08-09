"""top1_ctrl field/query grid sweep — isolated experiment.

Re-runs ``baselines/voyage_nano_field_sweep``'s exact 105-combo grid design
(every non-empty subset of positioning/background/lookingFor per side x
query on/off, plus a query-only seeker x the 7 candidate subsets) against a
different encoder: the fine-tuned ``top1_ctrl_001`` LoRA checkpoint
(``twotower_top1_optimised``) instead of frozen Voyage-4-nano — this
project's best fine-tuned model on the metric/population CLAUDE.md treats
as authoritative (holdout pair AUC 0.5974, hard-neg AUC 0.6121, MRR 0.5436;
the winning cell — micro-batch 6, single negative per anchor — from "Which
Lever Actually Worked?", `docs/twotower-ablation-verdict.md`).

A new encoder is a new experiment, not a new mode of an existing one, per
the "ML / data-science experiments" isolation rule in CLAUDE.md — hence a
new top-level package rather than a flag on ``voyage_nano_field_sweep``.
``fields.py``/``text.py`` are copied unchanged from that package (same grid
design); only ``encode.py`` (the encoder) is genuinely new, reusing
``twotower/eval.py``'s adapter-loading (``load_model_for_eval``,
``encode_role``) read-only and adding the disk-cache-by-name wrapper needed
for the same encoding-dedup trick. See
docs/twotower-top1-ctrl-field-sweep-experiment.md.
"""
