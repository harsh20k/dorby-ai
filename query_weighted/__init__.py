"""Does the searchQuery carry its weight in the seeker vector?

Motivation
----------
The seeker side of every model in this project is one string: eight tagged
profile fields with ``Search query: …`` appended. Measured on the 200 real
pairs, that query averages **218 characters against a 10,178-character seeker
string — about 2.1%**, roughly 55 tokens in 2,500.

That 2% is the only part of the text saying what the person wants *right now*;
the other 98% is biography. The existing ``*_no_query`` ablation already showed
removing it costs 20–26% of retrieval performance, so it is carrying weight far
out of proportion to its length — which is exactly the situation where it might
be starved.

This matters for a specific open problem. Fine-tuning improved recall@5 and
recall@10 by 10 queries each (47→57, 59→69 of 100) but recall@1 by one query
(18→19). The model reliably pulls the right person into the shortlist — topical
matching, which the profile bulk supports — and fails to convert to first place,
which is where the specific ask should decide. Query dilution is a plausible
mechanism. It has never been tested.

What this measures
------------------
Two questions, per the experiment's brief:

1. **Is the query helpful at all?** ``profile_only`` vs ``concat_baseline``.
2. **Does giving it more weight improve recall@1?** Two independent families:
   * **vector-level** — encode profile and query *separately*, combine as
     ``normalize(α·Q̂ + (1−α)·P̂)``. α sweeps 0→1, where α=0 is exactly
     ``profile_only`` and α=1 is exactly ``query_only``.
   * **text-level** — front-load the query before the profile, and repeat it
     k times, so the encoder itself sees it as more prominent.

The vector-level family is nearly free: encode profile-only and query-only once
each, and every α is then pure arithmetic. Only the text-level arms need fresh
encodes.

Isolation
---------
Own package, own artifacts dir, own Modal app and cache volume. Everything from
``baselines/`` — ``profile_to_text``, ``seeker_to_text``, ``candidate_to_text``,
``VoyageNanoEncoder``, and all of ``baselines.metrics`` — is imported and called
**unmodified**, so a number here is directly comparable to the published rows.
``eval_real_full`` supplies the 200-pair loader, also unmodified.

Validation gate
---------------
The ``concat_baseline`` arm is the current production seeker text. It must
reproduce frozen Voyage-4-nano's published all-200 numbers exactly — pair AUC
0.5593, MRR 0.3171, recall@1 0.1800. If it does not, the harness is wrong and no
other arm means anything. ``tests/test_query_weighted.py`` pins the text builder
against ``baselines.bert_frozen.text.seeker_to_text`` by exact string equality.
"""
