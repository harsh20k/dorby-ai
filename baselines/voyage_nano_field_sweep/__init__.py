"""Voyage-4-nano field/query grid sweep — isolated experiment.

A new isolated experiment, not an edit of ``baselines/voyage_nano_field_selected``
(the single seeker=positioning+lookingFor / candidate=positioning+background+
lookingFor configuration matching the LLM judge's focused prompt) — see the
"ML / data-science experiments" isolation rule in CLAUDE.md. That earlier
experiment tested one hand-picked field selection; this one grid-searches
every non-empty subset of the same three fields (``positioning``,
``background``, ``lookingFor``) independently for seeker and candidate, times
whether the searchQuery is included, to find the best-performing combination
rather than assuming the LLM judge's chosen fields also happen to be best for
an embedding model.

7 non-empty subsets per side x 7 x 2 query settings = 98 combinations, plus a
"query-only" seeker (no profile fields at all, searchQuery is the entire
seeker text) x the same 7 candidate subsets = 7 more, for 105 total, scored
on all 200 real pairs. Every combo gets the full metric suite this project
tracks elsewhere: pair ROC-AUC, best-F1, accuracy@0.5, hard/easy-neg AUC, and
retrieval MRR/mean-rank/median-rank/Recall@1,5,10/NDCG@1,5,10 — via
``baselines/metrics.py``'s ``pair_metrics``/``retrieval_metrics``/
``slice_metrics``, unmodified, so every row is directly comparable to every
other baseline in this project. See docs/voyage-nano-field-sweep-experiment.md.

Reuses ``baselines/voyage_nano/encode.py`` (``VoyageNanoEncoder``,
``cosine_scores``, ``pick_device``), ``baselines/llm_judge/real_pairs.py``
(real-pair loading), and ``baselines/metrics.py`` unmodified and read-only.
Only ``fields.py``, ``text.py``, ``eval.py``, ``sweep.py``, and
``modal_eval.py`` are new.
"""
