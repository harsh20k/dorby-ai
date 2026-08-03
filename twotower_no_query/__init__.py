"""Trains top1_ctrl's exact recipe with the search query removed from every
seeker text at training time, not just at eval time.

Two prior findings motivate this:

1. `twotower_top1_optimised/top1_ctrl_001` is the best fine-tune in the project
   (all-200 recall@1 0.19) — but like every other two-tower run in this repo,
   its seeker text was `profile + query` concatenated
   (`twotower.data.LabeledPair.seeker_text`).
2. `twotower_query_weighted/` found that swapping to a query-only or
   query-weighted seeker vector *at eval time*, on that same already-trained
   `top1_ctrl` adapter with no retraining, roughly doubles recall@1 (0.19 ->
   0.29-0.32).

That eval-time result leaves an open question: is the model just under-using
a query it barely learned to weight, or would training on profile-only text
from scratch do even better (or worse) than the eval-time trick? This package
answers that by actually retraining top1_ctrl's recipe on query-free rows.

Isolation
---------
Nothing under `twotower/`, `twotower_top1_optimised/`, or
`twotower_query_weighted/` is modified. `data.py`/`eval_dev.py` are
byte-identical copies of `twotower_top1_optimised`'s (module-path renamed
only, pinned by a numeric-agreement test); `config.py`/`train.py` reproduce
`top1_ctrl`'s exact hyperparameters (see config.py's docstring) so the only
experimental variable is the training text. `twotower.train`/`twotower.eval`
helpers and `baselines.metrics` are imported unchanged.

Training data: `scripts/build_rrf_multineg_triplets_no_query.py`, a deliberate
near-copy of `scripts/build_rrf_multineg_triplets.py` with one line changed
(`profile_to_text` instead of `seeker_to_text` for the anchor) — verified
row-for-row identical to the original k1 file on every field except `anchor`
(all 643 rows differ there; ids, positives, negatives match exactly).
"""
