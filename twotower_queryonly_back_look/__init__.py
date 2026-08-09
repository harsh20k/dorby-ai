"""Two-tower fine-tune: seeker = search query only, candidate = background + lookingFor.

Motivation
----------
The 105-way field/query sweep against the frozen `top1_ctrl` checkpoint
(`baselines/twotower_top1_ctrl_field_sweep/`, published as "top1_ctrl
field/query sweep") re-scored every non-empty seeker/candidate field subset
combination on all 200 real pairs and found:

- **Seeker side: fewer fields is monotonically better, all the way to
  zero.** Query-only seeker text beats every other seeker field-count
  average.
- **Best recall@1 of the whole 105-combo grid**: seeker = query only (no
  profile fields at all), candidate = `background` + `lookingFor`, query
  included — recall@1 0.28, recall@5 0.75, pair AUC 0.60.

That sweep only re-scored the existing `top1_ctrl` checkpoint frozen with
different input text — nobody has fine-tuned specifically on this
combination yet. This package does that: same 643-row `rrf_003`
population, same `top1_ctrl` recipe, only the seeker/candidate text
changed to match the sweep's recall@1 winner.

Caveat carried over from the sweep, repeated here because it directly
bears on this experiment: the sweep found the best combo (0.6395 pair AUC
by a different metric than recall@1) sits within run-to-run noise of the
10th-best (0.6167) on a single 200-pair population with no holdout check —
picking the best-scoring combo on the same population later reported on is
a search-then-report setup, not a validated production config. This
experiment does not resolve that caveat; it only tests whether the winning
combo *trains* well, which the frozen sweep could not answer.

One shared tower, not two — same reasoning as every fine-tune in this
project (`twotower_split/` already tested and lost with independent
towers); the shared tower already handles asymmetric seeker/candidate
content via Voyage's own query/document role prompts.

Text builders reused read-only, not duplicated: `query_weighted.text
.query_only` for the seeker side, `field_pairs_sweep.text
.background_lookingfor` for the candidate side. Row file built by
`scripts/build_rrf_multineg_triplets_queryonly_back_look.py`, verified
643 rows / 297 seekers / 0% padding — identical population to `top1_ctrl`'s
own row file.
"""
