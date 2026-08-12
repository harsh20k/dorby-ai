"""voyage_gemini_ctrl checkpoint-1130 field/query grid sweep — isolated experiment.

Re-runs the project's standard 105-combo grid design (every non-empty
subset of positioning/background/lookingFor per side x query on/off, plus
a query-only seeker x the 7 candidate subsets) against a *different
checkpoint* of the same training run already swept in
``baselines/twotower_voyage_gemini_ctrl_field_sweep``.

That earlier sweep used the adapter `top1_ctrl_001`'s training pulled from
the volume as ``.../voyage_gemini_ctrl_001/adapter`` — the checkpoint
`CorpusRecallDevEvaluator` selected as *best by dev recall@1*, at optimizer
step 452 (epoch 2.0) out of 1,130 total steps
(`docs/twotower-voyage-gemini-ctrl-experiment.md`). This package instead
uses ``checkpoints/checkpoint-1130`` — the raw final-epoch checkpoint,
saved but never selected, sitting on the same Modal volume
(``dorby-twotower-voyage-gemini-ctrl-checkpoints``). Confirmed identical
LoRA shape (rank 8, alpha 16, dropout 0.05, q/k/v/o_proj) to every other
checkpoint in this project — same base model, different point in training.

Every other custom-loop two-tower run in this project except
`queryonly_back_look_001` showed dev recall@1 peak early (epoch 1-2) and
decay afterward — the reason `top1_ctrl`'s recipe always selects by
recall@1 rather than shipping the final epoch (`docs/possible-bugs.md` #2).
This experiment asks whether that decay, if present here too, also shows
up in the *field/query sweep* results — does the checkpoint the training
pipeline discarded do better, worse, or about the same as the one it kept,
once every text representation is tried instead of just the training one?

A different checkpoint is a new experiment, not a new mode of an existing
one, per the "ML / data-science experiments" isolation rule in CLAUDE.md —
hence a new top-level package rather than a flag on
``twotower_voyage_gemini_ctrl_field_sweep``. ``fields.py``/``text.py`` are
copied unchanged (same grid design); only ``encode.py`` changes, pointing
at ``checkpoint-1130`` instead of the selected adapter. See
docs/twotower-voyage-gemini-ctrl-ckpt1130-field-sweep-experiment.md.
"""
