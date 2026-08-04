"""Blend the query with the seeker's best-matching lookingFor section, not the whole profile.

Motivation
----------
Two prior findings point at the same gap from different directions:

* ``nomad_drift/`` calibrated the query/profile alpha blend on rrf_003 and found
  pure query (alpha=1.0) wins retrieval and hard-negative AUC but *loses* a bit
  of overall pair AUC to a partial blend (alpha=0.6) — some profile content
  still helps, dropping it all costs something.
* ``baselines/voyage_nano_sectioned/`` (see ``docs/lookingfor-sectioning-findings.md``)
  found that splitting a **seeker's own** ``lookingFor`` into per-ask sections and
  scoring by the single best-matching section (not the whole field) lifts pair
  AUC 0.579->0.596 and top-1 retrieval 27.6%->34.5% on the 69-pair holdout — a
  seeker's asks are independent threads, and any one query is only ever about
  one of them, so the *other* sections are pure dilution, not signal.

Neither experiment tried the other's mechanism. This package does: instead of
blending the query with the whole profile vector (``nomad_drift``) or scoring by
max-over-sections against every candidate (``voyage_nano_sectioned``, an O(sections
x candidates) multi-vector approach with no single seeker vector), it picks
**one** section per query — whichever has the highest cosine to the query vector
alone, an O(sections) comparison — and blends only that section with the query,
producing a single vector suitable for ordinary ANN retrieval. The hope: recover
some of the pair-AUC ground pure query gives up, without paying the whole
profile's dilution cost.

rrf_003 is an unusually good fit for calibrating this: every query_key in it was
generated *for* one specific lookingFor section
(``synth_pipeline/pairing_rrf/sections.py::query_targets``), so each pair record
carries a ground-truth section index. That means calibration here can check not
just "does the blend help" but "does cosine-similarity selection actually pick
the section the query was written for" — a mechanism check the frozen-baseline
sectioning experiment never had available.

Isolation
---------
Own package, own Modal app, own artifacts dir. Reuses, unmodified:
``nomad_drift.calibrate`` (rrf_003 loading, the alpha-blend arithmetic),
``synth_pipeline.pairing_rrf.sections.seeker_vectors`` (the canonical seeker
section-splitter, shared with ``baselines/voyage_nano_sectioned`` so the two
cannot drift), and ``query_weighted.eval.encode_everything`` / ``score_arm``
(the real-200 scoring harness) — this package's custom seeker vectors are scored
by calling ``score_arm`` directly with a substituted seeker matrix, so the
metrics are computed by the exact function that produced every other row in
this project's tables, not a reimplementation of it.
"""
