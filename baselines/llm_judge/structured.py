"""Parse + aggregate the ``structured_cot`` judge variant's per-aspect scores.

The ``naive``/``calibrated`` variants ask for a single yes/no straight away.
This variant instead asks the model to score six independent aspects (see
``prompts/structured_cot.md``) and only then derive a verdict — the test this
experiment runs is whether forcing that decomposition before an answer moves
pair AUC versus asking for the answer directly.

Aggregation happens in code, not in the model's own arithmetic: the model
computes ``weighted_score`` too (as a check the prompt asks for explicitly),
but the value actually scored is recomputed here from the per-aspect scores
and the *canonical* fixed weights below, so a model that quietly reweights
one aspect to swing its own answer cannot move the number that gets measured.
"""

from __future__ import annotations

from typing import Any

# (name, weight) — must match prompts/structured_cot.md exactly, weights sum to 1.0.
CANONICAL_ASPECTS: tuple[tuple[str, float], ...] = (
    ("location_availability", 0.15),
    ("ask_offer_alignment", 0.25),
    ("skill_domain_evidence", 0.20),
    ("seniority_stage_fit", 0.15),
    ("domain_industry_fit", 0.15),
    ("practical_constraints", 0.10),
)
CANONICAL_WEIGHTS: dict[str, float] = dict(CANONICAL_ASPECTS)
CANONICAL_NAMES = frozenset(CANONICAL_WEIGHTS)

assert abs(sum(CANONICAL_WEIGHTS.values()) - 1.0) < 1e-9


def parse_structured_verdict(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate + score a structured_cot response into the shared verdict shape.

    Returns the same ``{match, confidence, reasoning}`` keys every other verdict
    uses (so ``decision_metrics``/``calibration_buckets``/``verdict_to_score``
    all work unmodified) plus ``aspects`` and ``weighted_score`` for the raw
    per-aspect record.
    """
    aspects = raw.get("aspects")
    if not isinstance(aspects, list) or len(aspects) != len(CANONICAL_ASPECTS):
        raise ValueError(f"'aspects' must be a list of {len(CANONICAL_ASPECTS)} items, got {aspects!r}")

    scores: dict[str, float] = {}
    evidence: dict[str, str] = {}
    for item in aspects:
        if not isinstance(item, dict):
            raise ValueError(f"each aspect must be an object, got {item!r}")
        name = item.get("name")
        if name not in CANONICAL_NAMES:
            raise ValueError(f"unknown aspect name {name!r}; expected one of {sorted(CANONICAL_NAMES)}")
        if name in scores:
            raise ValueError(f"aspect {name!r} appears more than once")
        try:
            score = float(item.get("score"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"aspect {name!r} has non-numeric score: {item.get('score')!r}") from exc
        if not 0.0 <= score <= 5.0:
            raise ValueError(f"aspect {name!r} score out of range 0-5: {score!r}")
        scores[name] = score
        ev = item.get("evidence")
        evidence[name] = ev if isinstance(ev, str) and ev.strip() else ""

    missing = CANONICAL_NAMES - set(scores)
    if missing:
        raise ValueError(f"missing aspect(s): {sorted(missing)}")

    # Recomputed from canonical weights, deliberately ignoring whatever weight
    # the model echoed back in each aspect object or in its own top-level
    # "weighted_score" — see module docstring.
    weighted_score = sum(CANONICAL_WEIGHTS[name] * (scores[name] / 5.0) for name in CANONICAL_WEIGHTS)
    weighted_score = max(0.0, min(1.0, weighted_score))

    match = "yes" if weighted_score >= 0.5 else "no"
    confidence = min(100.0, abs(weighted_score - 0.5) * 200.0)

    synthesis = raw.get("synthesis")
    reasoning = synthesis if isinstance(synthesis, str) and synthesis.strip() else (
        "; ".join(f"{name}={scores[name]:.1f}: {evidence[name]}" for name in CANONICAL_WEIGHTS)
    )

    return {
        "match": match,
        "confidence": round(confidence, 2),
        "reasoning": reasoning,
        "aspects": [
            {"name": name, "weight": CANONICAL_WEIGHTS[name], "score": scores[name], "evidence": evidence[name]}
            for name in CANONICAL_WEIGHTS
        ],
        "weighted_score": round(weighted_score, 4),
    }
