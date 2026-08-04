"""Calibrate the query/profile blend weight on synthetic data, confirm on real pairs.

Motivation
----------
``query_weighted/`` found that encoding a seeker's profile and search query as
**separate** vectors and blending them — ``normalize(alpha*query + (1-alpha)*
profile)`` — nearly doubles recall@1 on frozen voyage-4-nano (0.18 -> 0.30, all
200 real pairs). It left one thing unresolved: alpha was read off a 9-point grid
evaluated on the same 200 pairs the finding is reported on, so 0.6 vs 0.7 is not
distinguishable from noise, and the grid was never fit on anything held out.

This package fits alpha on a *different, much larger, and separate* population —
the 1000-profile synthetic batch ``rrf_003`` (923 unique profiles, 2,619
judge-labeled pairs from ``synth_pipeline/pairing_rrf/``) — then reports the
winning alpha's performance on all 200 real pairs, a population it never saw
during calibration. That is the train/test split the original experiment was
missing, using a population disjoint from the reporting one instead of a
same-population held-out slice.

Two metrics drive calibration, both corpus-free (no circularity from a shared
retrieval corpus, matching the "hard-negative AUC is the clean signal" caveat in
``docs/query-weighted-encoding-experiment.md``):

* **pair AUC** — cosine(seeker', candidate) vs pos/neg label, over all 2,619
  synthetic pairs.
* **shortlist top-1 accuracy** — within each query's judged shortlist (an RRF
  top-5 of dense+lexical retrieval, some candidates pos, some neg), does the
  highest-scoring candidate carry the pos label? Restricted to query_keys that
  have at least one of each label, since a shortlist with only one class has
  nothing to discriminate. This is deliberately *not* called "recall@1" — it is
  a within-shortlist ranking accuracy over ~5 candidates, not a full-corpus
  retrieval metric like the real-200 report below.

Isolation
---------
Own package, own artifacts dir, own Modal app. Everything reused —
``query_weighted.text``'s builders, ``query_weighted.eval.combine`` /
``run_all_arms``, ``baselines.metrics.pair_metrics``,
``baselines.voyage_nano.encode.VoyageNanoEncoder`` — is imported unmodified. The
real-200 report is literally ``query_weighted.eval.run_all_arms`` called with
this package's calibrated alpha folded into its alpha grid, so those numbers are
directly comparable to the published ``query_weighted`` table, not a reimplementation
of it.

Labels caveat
-------------
``rrf_003``'s pos/neg labels come from an LLM judge over synthetic profiles, not
real accept/decline outcomes (see ``docs/rrf-pairing-pipeline.md``). They are
good enough to pick a scalar hyperparameter from a much bigger sample than 200
pairs affords, but the number that matters is still the real-200 report, which
this package always runs and treats as the headline result.
"""
