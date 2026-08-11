"""Do positioning + background carry the seeker's identity on their own?

Motivation
----------
The field-isolation experiment (``baselines/voyage_nano_field_isolation``,
logged in ``docs/experiment-graphs-index.md``) embedded each profile field
completely alone and measured how well it re-identifies its own owner.
Three fields came back clearly person-specific: ``positioning`` (0.890
own-owner cosine vs 0.549 across strangers), ``background`` (0.850 vs
0.580), and ``lookingFor`` (0.852 vs 0.595). Everything else —
``locationAvailability``, ``personalPreferences``,
``meetingAndSchedulingPreferences`` — read as generic scheduling
boilerplate (0.37-0.45 own-owner, *below* their cross-person baseline).

Separately, ``query_weighted`` established that the full 8-field profile
dilutes the search query rather than helping it: ``query_only`` beats
``concat_baseline`` on every retrieval metric.

This experiment tests a middle ground on the seeker side: keep only the two
identity-carrying fields at a time, drop the query and every other field
entirely, and see whether that beats the noisier full-profile baseline. Three
arms cover all three pairs among the three identity fields:

- ``pos_background``       = positioning + background
- ``pos_lookingfor``       = positioning + lookingFor
- ``background_lookingfor``= background + lookingFor

The candidate side is untouched everywhere in this project — only the seeker
side changes.

Isolation
---------
New top-level package. ``field_pairs_sweep/text.py`` builds the three arms
directly from ``baselines.bert_frozen.text.PROFILE_FIELDS`` tagging (same
``"field: value"`` format, same ``\\n\\n`` join) but is not a copy of any
existing builder — no prior experiment concatenates two arbitrary fields.
Everything else — ``VoyageNanoEncoder``, ``baselines.metrics``,
``eval_real_full``'s 200-pair loader — is imported and called unmodified,
so a number here is directly comparable to the published
``query_weighted`` rows (``profile_only`` 0.5593 AUC / 0.3171 MRR baseline,
``query_only`` as the current best pure-text arm).

Free eval-time sweep first: three arms, frozen voyage-4-nano, no training.
Only if an arm beats ``query_only`` on the all-200 population is a trained
adapter worth building.
"""
