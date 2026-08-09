"""queryonly_back_look field/query grid sweep — isolated experiment.

Re-runs ``baselines/voyage_nano_field_sweep``'s exact 105-combo grid design
(every non-empty subset of positioning/background/lookingFor per side x
query on/off, plus a query-only seeker x the 7 candidate subsets) against a
different encoder: the fine-tuned ``queryonly_back_look_001`` LoRA checkpoint
(``twotower_queryonly_back_look``) instead of frozen Voyage-4-nano or
``top1_ctrl``. As of this experiment, ``queryonly_back_look_001`` is the
**new best two-tower fine-tune in the project on every tracked metric**
(all-200: pair AUC 0.5983, hard-neg AUC 0.6564, MRR 0.4791, R@1 0.30 — vs
``top1_ctrl``'s 0.5683 / 0.5484 / 0.3550 / 0.19), trained on the field/query
sweep's own recall@1-best text representation (seeker = search query only,
candidate = background+lookingFor) found by
``baselines/twotower_top1_ctrl_field_sweep`` and confirmed to train well in
``docs/twotower-queryonly-back-look-experiment.md``.

Running the *generalized* 105-combo grid against this checkpoint (not just
its own training combo) asks a different question than that doc did: does
fine-tuning on one specific text representation help or hurt the model's
embeddings for every *other* representation too, the same question already
asked of ``top1_ctrl`` by ``twotower_top1_ctrl_field_sweep``.

A new encoder is a new experiment, not a new mode of an existing one, per
the "ML / data-science experiments" isolation rule in CLAUDE.md — hence a
new top-level package rather than a flag on an existing sweep.
``fields.py``/``text.py`` are copied unchanged from
``twotower_top1_ctrl_field_sweep`` (same grid design); only ``encode.py``
(the encoder) is genuinely new, pointing at the new adapter checkpoint
while reusing ``twotower/eval.py``'s adapter-loading
(``load_model_for_eval``, ``encode_role``) read-only. See
docs/twotower-queryonly-back-look-field-sweep-experiment.md.
"""
