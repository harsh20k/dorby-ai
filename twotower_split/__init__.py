"""Genuinely split two-tower: separate LoRA adapters for query vs. candidate,
instead of one shared model reading different input text.

Every prior two-tower experiment in this project (`top1_ctrl`,
`twotower_no_query/`, `twotower_query_only/`) used ONE model for both sides —
same weights, only the input text (and a role prompt) differed. This package
tests the architectural change itself: a dedicated query-tower adapter and a
dedicated candidate-tower adapter, trained jointly with a shared contrastive
loss, so each can specialize for its own job instead of being pulled in two
directions by one shared set of weights.

Query-side text is `query_only` (the strongest seeker representation found
so far — see `twotower_query_only/`); candidate-side text is unchanged
`candidate_to_text`. Same rows as `twotower_query_only/`'s already-built
`rrf_003_multineg_k1_query_only.json` — reused directly, not regenerated.

Because the two adapters must be routed to different data columns per
forward pass, `SentenceTransformerTrainer` (which assumes one shared model)
can't express this — `train.py` is a from-scratch training loop reusing
`twotower.train`'s generic, model-agnostic helpers (`build_model`,
`add_lora_adapter`) read-only, plus a hand-written
MultipleNegativesRankingLoss-equivalent (in-batch + one explicit hard
negative, scale 20 — same formula the library loss uses).

Nothing under `twotower/`, `twotower_query_only/`, `twotower_no_query/`, or
`twotower_top1_optimised/` is modified.
"""
