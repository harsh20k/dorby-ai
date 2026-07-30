"""Aggregation strategies for collapsing per-section scores into a per-record score.

A seeker's ``lookingFor`` field is split into N sections (see
``text.py::seeker_to_sectioned_texts``); each section is embedded separately and
scored against every candidate, producing an ``(n_sections, n_candidates)`` score
matrix. This module collapses that matrix, one contiguous block of rows at a
time (grouped by ``offsets``, the same per-record offset list produced by
``eval_seeker.py::build_sectioned_seekers``), into an ``(n_records, n_candidates)``
score matrix.

Modes split into two families, which encode two different theories of what a
match *is*. The distinction is the point of the ``min``/``softmin``/``noisy_or``
additions, so it is worth stating plainly:

**Relevance-shaped** ("does any ask fit?") — a candidate is good if it strongly
matches at least one of the seeker's asks. The seeker's other, irrelevant asks
should not count against them.

  - ``max``: the original hard max-pool — a record's score against a candidate
    is its single best-matching section. Improves pair AUC/MRR/top-1 over the
    unsectioned baseline but loses some Recall@10, the theory being that a
    candidate's rank among a seeker's *other* sections gets thrown away
    entirely.
  - ``topk_mean``: mean of the ``topk`` highest-scoring sections per candidate
    (falls back to averaging all sections if the seeker has fewer than
    ``topk``). A softer middle ground between max and a full average.
  - ``softmax``: temperature-weighted average over all sections per candidate
    — every section contributes, but better-matching sections dominate the
    weighted average more as ``temperature`` shrinks toward 0 (approaching
    hard max) or contribute closer to equally as it grows. This is exactly the
    temperature-sharpened gate ``g_m(x) = exp(a_m/T) / Σ exp(a_j/T)``, with the
    gate placed over *sections* rather than over experts.

**Veto-shaped** ("is any ask badly violated?") — a candidate is bad if it fails
on even one of the seeker's asks, however well it does on the others. This is
the "one dealbreaker sinks the intro" theory. Untested before now; the reason
it matters is that averaging independent sub-scores regresses toward the
decision boundary (see the ``structured_cot`` result in
``docs/llm-judge-experiment.md``), and a veto does not.

  - ``min``: hard veto — a record's score is its *worst*-matching section.
  - ``softmin``: temperature-weighted average biased toward the worst sections;
    the exact mirror of ``softmax`` (weights use ``-group/temperature``). At
    ``temperature`` → 0 it approaches ``min``; large ``temperature`` approaches
    ``mean``.
  - ``noisy_or``: probabilistic veto. Cosines are mapped to pseudo-probabilities
    by the **fixed, data-independent** map ``p = (1 + cos) / 2`` (cosine is
    bounded on [-1, 1], so this needs no statistic estimated from the data),
    then combined as a **length-normalized noisy-AND**: ``exp(mean(log p))``,
    the geometric mean. Note the naming: noisy-OR over *failure* probabilities
    is algebraically the product over *match* probabilities, so this is the
    noisy-OR-of-dealbreakers the MoE discussion asks for, written in match
    space. The length normalization is not optional — a raw product would give
    a seeker with 126 sections a score of ~0 no matter how well they matched,
    making section count dominate everything else.

    The fixed map is load-bearing, not a stylistic choice. An earlier version
    rescaled by the min/max of the score matrix it was passed. Because
    ``eval_seeker._score_with_agg`` aggregates the positive and negative
    matrices in *separate* calls, that made the transform label-dependent —
    positives were rescaled by the positive matrix's range and negatives by the
    negative matrix's, so an identical cosine became a higher probability on the
    positive side. It reported pair AUC 0.8500 against a ~0.60 field. Every
    other mode here is computed strictly within one record's group of sections
    and so cannot express this bug; ``noisy_or`` was the first mode to reach for
    a statistic outside the group. ``tests/test_section_aggregation.py``
    pins the invariant (a record's score must not depend on which other records
    share its batch).

**Reference point**

  - ``mean``: plain average over all sections. Not proposed as a good scorer —
    it is the control that shows what full averaging costs, i.e. the
    aggregation shape that lost as ``structured_cot``.
"""

from __future__ import annotations

import numpy as np


AGG_MODES = ("max", "topk_mean", "softmax", "mean", "min", "softmin", "noisy_or")

#: Which theory of matching each mode encodes — used by reporting/tests so the
#: two families never get silently mixed up in a results table.
AGG_FAMILY = {
    "max": "relevance",
    "topk_mean": "relevance",
    "softmax": "relevance",
    "mean": "average",
    "min": "veto",
    "softmin": "veto",
    "noisy_or": "veto",
}


def aggregate_sections(
    scores: np.ndarray,
    offsets: list[int],
    mode: str = "max",
    topk: int = 2,
    temperature: float = 0.05,
    prob_floor: float = 1e-3,
) -> np.ndarray:
    """Collapse an (n_sections, n_candidates) score matrix into (n_groups, n_candidates).

    ``offsets`` is the full per-record offset list (length n_groups + 1); group i
    spans rows ``offsets[i]:offsets[i + 1]``.

    ``prob_floor`` only affects ``noisy_or``: it clips the globally-rescaled
    pseudo-probabilities away from exactly 0 and 1 so ``log p`` stays finite.
    """
    if mode not in AGG_MODES:
        raise ValueError(f"unknown mode: {mode} (expected one of {AGG_MODES})")

    n_groups = len(offsets) - 1
    out = np.empty((n_groups, scores.shape[1]), dtype=scores.dtype)

    log_probs: np.ndarray | None = None
    if mode == "noisy_or":
        # Fixed cosine -> probability map. MUST NOT depend on any statistic of
        # `scores`: this function is called separately for the positive and the
        # negative matrices, so a data-derived rescale is label leakage. See the
        # module docstring.
        probs = (1.0 + scores) / 2.0
        log_probs = np.log(np.clip(probs, prob_floor, 1.0 - prob_floor))

    for i in range(n_groups):
        group = scores[offsets[i] : offsets[i + 1]]
        if mode == "max":
            out[i] = group.max(axis=0)
        elif mode == "topk_mean":
            k = min(topk, group.shape[0])
            out[i] = np.sort(group, axis=0)[-k:].mean(axis=0)
        elif mode == "softmax":
            w = np.exp((group - group.max(axis=0, keepdims=True)) / temperature)
            w = w / w.sum(axis=0, keepdims=True)
            out[i] = (w * group).sum(axis=0)
        elif mode == "mean":
            out[i] = group.mean(axis=0)
        elif mode == "min":
            out[i] = group.min(axis=0)
        elif mode == "softmin":
            # Mirror of softmax: weight mass moves to the *lowest*-scoring
            # sections. Subtract the per-candidate min (not max) for stability.
            w = np.exp(-(group - group.min(axis=0, keepdims=True)) / temperature)
            w = w / w.sum(axis=0, keepdims=True)
            out[i] = (w * group).sum(axis=0)
        elif mode == "noisy_or":
            assert log_probs is not None
            lp = log_probs[offsets[i] : offsets[i + 1]]
            # Geometric mean = length-normalized product. Equivalent to
            # noisy-OR over per-section failure probabilities.
            out[i] = np.exp(lp.mean(axis=0))

    return out
