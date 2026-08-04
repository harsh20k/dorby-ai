"""Trains top1_ctrl's exact recipe with the seeker side built from the search
query alone — no profile text anywhere in training.

Third leg of a three-way seeker-text comparison:

1. `twotower_top1_optimised/top1_ctrl` — trained on profile+query concatenated.
2. `twotower_no_query/` — trained on profile only. Found training condition
   barely moves profile-only eval performance: `no_query_001` landed within
   noise of `top1_ctrl`'s own eval-time profile-only swap (0.13 R@1 both).
3. This package — trained on query only. Question: does training on
   query-only text beat `top1_ctrl`'s eval-time query-only swap (R@1 0.32,
   the best fine-tuned-model retrieval number on record), or does it land
   close to it the same way (2) did?

Isolation: nothing under `twotower/`, `twotower_top1_optimised/`,
`twotower_no_query/`, or `twotower_query_weighted/` is modified.
`data.py`/`eval_dev.py` are renamed-import copies of `twotower_no_query`'s
(themselves copies of the ablation's); `config.py`/`train.py` reproduce
`top1_ctrl`'s exact hyperparameters. Training data:
`scripts/build_rrf_multineg_triplets_query_only.py`, a near-copy of the
no-query script's sibling, reusing `query_weighted.text.query_only` read-only
for the anchor text (its empty-query fallback logic, not reimplemented) —
verified row-for-row identical to the original k1 file except `anchor`.

Evaluation is matched-distribution **from the start** this time — the bug in
`twotower_no_query/`'s first pass (scored on text the model never trained on;
see `docs/possible-bugs.md` #5) is not repeated here. `modal_eval_matched.py`
scores this adapter on `query_only` seeker text via
`twotower_query_weighted.eval`'s already-published `query_only` path,
read-only, matching training from the first eval run.
"""
