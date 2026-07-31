"""Per-section mixture-of-experts re-ranker with learned attention pooling.

The unit of prediction is a **(pair, section) row**, not a pair: a seeker's
``lookingFor`` field is split into its individual asks, each ask is scored
against the candidate separately, and the per-ask verdicts are pooled back into
one pair verdict by a learned attention head.

Isolated per the experiment rule in CLAUDE.md — nothing in ``moe_reranker/``,
``moe_rrf/``, ``baselines/`` or ``twotower/`` is modified. Shared code is
imported read-only through its public API.
"""
