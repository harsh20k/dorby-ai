"""voyage_gemini_ctrl field/query grid sweep — isolated experiment.

Re-runs the project's standard 105-combo grid design (every non-empty
subset of positioning/background/lookingFor per side x query on/off, plus
a query-only seeker x the 7 candidate subsets) against a third fine-tuned
encoder: the ``voyage_gemini_ctrl_001`` LoRA checkpoint
(``twotower_voyage_gemini_ctrl``) instead of frozen Voyage-4-nano,
``top1_ctrl``, or ``queryonly_back_look_001``.

``voyage_gemini_ctrl_001`` is ``top1_ctrl``'s exact recipe retrained on a
bigger, newer (but unreviewed, measurably leakier) synthetic batch
(``pairing_voyage_gemini/smoke_test_002``, 3,008 rows vs. `top1_ctrl`'s 643)
with the same full-profile text on both sides. All-200: pair AUC 0.6081 —
the best pair AUC of any fine-tune in the project — but it does not beat
`queryonly_back_look_001` on hard-neg AUC (0.6264 vs 0.6564), MRR (0.4506 vs
0.4791), R@1 (0.26 vs 0.30), or R@10 (0.81 vs 0.86); see
`docs/twotower-voyage-gemini-ctrl-experiment.md`, which explicitly flags the
field/query sweep as the natural next step for this checkpoint — this
package is that next step.

A new encoder is a new experiment, not a new mode of an existing one, per
the "ML / data-science experiments" isolation rule in CLAUDE.md — hence a
new top-level package rather than a flag on an existing sweep. ``fields.py``
/``text.py`` are copied unchanged from ``twotower_queryonly_back_look_field
_sweep`` (same grid design); only ``encode.py`` (the encoder) is genuinely
new, pointing at this checkpoint while reusing ``twotower/eval.py``'s
adapter-loading (``load_model_for_eval``, ``encode_role``) read-only. See
docs/twotower-voyage-gemini-ctrl-field-sweep-experiment.md.
"""
