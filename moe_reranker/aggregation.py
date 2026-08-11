"""Section-score aggregation shapes, owned by the MoE experiment.

**Why this is here and not in ``baselines/voyage_nano_sectioned/aggregate.py``.**
The three relevance-shaped modes (``max``, ``topk_mean``, ``softmax``) are
reimplemented here rather than imported, so this experiment can add veto-shaped
modes without editing a module the sectioning baseline owns. The shared file is
left exactly as that experiment wrote it; the duplication is deliberate
isolation, not an oversight. If the shared implementations ever change,
``tests/test_moe_aggregation.py::test_matches_shared_baseline_on_shared_modes``
fails and tells you.

The two families encode two different theories of what a match *is*:

**Relevance-shaped** ("does any ask fit?") — a candidate is good if it strongly
matches at least one of the seeker's asks; the seeker's other, irrelevant asks
should not count against them.

  - ``max``       hard max-pool: the single best-matching section.
  - ``topk_mean`` mean of the ``topk`` best sections.
  - ``softmax``   temperature-weighted average. This is the professor's Idea 1,
                  ``g_m(x) = exp(a_m/T) / Σ exp(a_j/T)``, with the gate placed
                  over *sections* rather than over experts.

**Veto-shaped** ("is any ask badly violated?") — a candidate is bad if it fails
even one ask, however well it does on the others. The "one dealbreaker sinks the
intro" theory. It matters because averaging independent sub-scores regresses
toward the decision boundary (the ``structured_cot`` result in
``docs/llm-judge-experiment.md``) and a veto does not.

  - ``min``      hard veto: the worst-matching section.
  - ``softmin``  exact mirror of ``softmax`` (weights use ``-group/temperature``).
  - ``noisy_or`` probabilistic veto. Cosines map to pseudo-probabilities by the
                 **fixed, data-independent** ``p = (1 + cos) / 2`` (cosine is
                 bounded on [-1, 1], so no statistic is estimated from the data),
                 combined as a length-normalized noisy-AND ``exp(mean(log p))``.
                 Noisy-OR over per-section *failure* probabilities is
                 algebraically the product over *match* probabilities, so this is
                 the noisy-OR-of-dealbreakers written in match space. The length
                 normalization is not optional: a raw product would score a
                 seeker with 126 asks at ~0 regardless of fit.

**Reference point**

  - ``mean``     plain average. Not a proposal — the control showing what full
                 averaging costs.

**A bug this module is shaped to prevent.** An earlier ``noisy_or`` rescaled by
the min/max of the score matrix it was passed. Callers aggregate the positive and
negative matrices in *separate* calls, so that made the transform label-dependent
— positives rescaled by the positive range, negatives by the negative range — and
it reported pair AUC 0.8500 against a ~0.60 field. Every mode here must therefore
compute strictly within one record's own group of sections.
``tests/test_moe_aggregation.py::test_record_score_is_batch_independent`` pins it.

Result on the real 69-pair holdout (``scripts/compare_section_aggregation.py``):
veto-shaped lost at every level, monotonically, on both pair AUC and hard-negative
AUC. Best was ``softmax(T=0.05)`` at 0.5983; worst was hard ``min`` at 0.5836. No
single gap is significant (paired bootstrap CI straddles zero); the ordered ladder
is what carries the signal.
"""

from __future__ import annotations

import numpy as np

AGG_MODES = ("max", "topk_mean", "softmax", "mean", "min", "softmin", "noisy_or")

#: Which theory of matching each mode encodes, so results tables never silently
#: mix the two families.
AGG_FAMILY = {
    "max": "relevance",
    "topk_mean": "relevance",
    "softmax": "relevance",
    "mean": "average",
    "min": "veto",
    "softmin": "veto",
    "noisy_or": "veto",
}

#: Modes that also exist in `baselines/voyage_nano_sectioned/aggregate.py` and
#: must stay numerically identical to it.
SHARED_MODES = ("max", "topk_mean", "softmax")


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

    ``prob_floor`` affects ``noisy_or`` only: it clips the pseudo-probabilities
    away from exactly 0 and 1 so ``log p`` stays finite.
    """
    if mode not in AGG_MODES:
        raise ValueError(f"unknown mode: {mode} (expected one of {AGG_MODES})")

    n_groups = len(offsets) - 1
    out = np.empty((n_groups, scores.shape[1]), dtype=scores.dtype)

    log_probs: np.ndarray | None = None
    if mode == "noisy_or":
        # Fixed cosine -> probability map. MUST NOT depend on any statistic of
        # `scores`; see the module docstring for what happens when it does.
        log_probs = np.log(np.clip((1.0 + scores) / 2.0, prob_floor, 1.0 - prob_floor))

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
            out[i] = np.exp(lp.mean(axis=0))

    return out
