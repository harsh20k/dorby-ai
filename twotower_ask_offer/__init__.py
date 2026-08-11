"""Two-tower reciprocal fine-tune: separate Ask/Offer LoRA adapters trained
jointly on S = s_fwd + lambda*s_rev, instead of one shared tower trained on
s_fwd alone.

Design doc: docs/html/reciprocal-two-tower-training-plan.html
(https://claude.ai/code/artifact/373941ab-0755-4948-af61-9eba1fd34bc8).
Builds on baselines/reciprocal_static/ (the zero-training version of this
score, already run on real data) and reuses its text.py split read-only.

New top-level package per the experiment-isolation rule: nothing under
twotower/, baselines/reciprocal_static/, or twotower_voyage_gemini_ctrl/ is
modified. Training data is frozen from voyage_gemini_ctrl's source batch via
import_rows.py (same 3008 rows, same ids) so this experiment differs from
that one in exactly one place — two towers + the reciprocal loss — not in
the training population too.
"""
