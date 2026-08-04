"""One shared tower, seeker side split into pieces (search query, `lookingFor`,
`positioning`), combined by a small learned gate instead of one flat string
or one fixed alpha blend.

Motivated by `query_weighted/`'s finding that a *fixed* alpha blend
(profile vs. query) beats plain concatenation, and by the "decompose the
profile into weighted sections" idea flagged as still-open after
`twotower_no_query/` and `twotower_query_only/` ruled out "training on one
whole-text variant vs. another." This package tests a learned, per-seeker
combination instead of one global alpha: `gate.py`'s `FieldGate` takes the
three piece embeddings, computes one softmax weight per piece from a small
linear layer, and returns the weighted sum — the model decides for itself,
per seeker, how much to weight the query vs. each profile field.

Single shared tower (one LoRA adapter) for every text, seeker pieces and
candidate alike — unlike `twotower_split/`, which found two independent
towers actively hurt. Candidate side is unchanged `candidate_to_text`; the
novelty is scoped to the seeker side only.

Custom training loop for the same reason as `twotower_split/train.py`:
`SentenceTransformerTrainer` cannot express "encode three pieces, then run
them through an extra trainable module before the loss." Reuses
`twotower.train`'s generic helpers (`build_model`, `add_lora_adapter`)
read-only.

Nothing under `twotower/`, `twotower_top1_optimised/`, `twotower_no_query/`,
`twotower_query_only/`, or `twotower_split/` is modified.
"""
