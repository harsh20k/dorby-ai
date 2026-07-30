"""Aggregation strategies for collapsing per-section scores into a per-record score.

A seeker's ``lookingFor`` field is split into N sections (see
``text.py::seeker_to_sectioned_texts``); each section is embedded separately and
scored against every candidate, producing an ``(n_sections, n_candidates)`` score
matrix. This module collapses that matrix, one contiguous block of rows at a
time (grouped by ``offsets``, the same per-record offset list produced by
``eval_seeker.py::build_sectioned_seekers``), into an ``(n_records, n_candidates)``
score matrix.

Three modes:
  - ``max``: the original hard max-pool (``np.maximum.reduceat`` equivalent) —
    a record's score against a candidate is its single best-matching section.
    Improves pair AUC/MRR/top-1 over the unsectioned baseline but loses some
    Recall@10, the theory being that a candidate's rank among a seeker's
    *other* sections gets thrown away entirely.
  - ``topk_mean``: mean of the ``topk`` highest-scoring sections per candidate
    (falls back to averaging all sections if the seeker has fewer than
    ``topk``). A softer middle ground between max and a full average.
  - ``softmax``: temperature-weighted average over all sections per candidate
    — every section contributes, but better-matching sections dominate the
    weighted average more as ``temperature`` shrinks toward 0 (approaching
    hard max) or contribute closer to equally as it grows.
"""

from __future__ import annotations

import numpy as np


def aggregate_sections(
    scores: np.ndarray,
    offsets: list[int],
    mode: str = "max",
    topk: int = 2,
    temperature: float = 0.05,
) -> np.ndarray:
    """Collapse an (n_sections, n_candidates) score matrix into (n_groups, n_candidates).

    ``offsets`` is the full per-record offset list (length n_groups + 1); group i
    spans rows ``offsets[i]:offsets[i + 1]``.
    """
    n_groups = len(offsets) - 1
    out = np.empty((n_groups, scores.shape[1]), dtype=scores.dtype)
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
        else:
            raise ValueError(f"unknown mode: {mode}")
    return out
