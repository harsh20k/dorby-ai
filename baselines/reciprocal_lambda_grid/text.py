"""Look-for vs. background text split for the lambda-grid-sweep experiment.

Deliberate duplicate of baselines/reciprocal_static_rrf003/text.py (experiment
isolation rule — each experiment's numbers must stay reproducible from its own
code). Same field choice as that package: bg_text = positioning + background
only. look_text / seeker_look_text are unchanged from the original
baselines/reciprocal_static/text.py.
"""

from __future__ import annotations

from typing import Any, Mapping

BG_FIELDS = ("positioning", "background")


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def bg_text(profile: Mapping[str, Any]) -> str:
    """Background text v_i: positioning + background fields only, tagged."""
    parts: list[str] = []
    for field in BG_FIELDS:
        value = profile.get(field)
        if _nonempty(value):
            parts.append(f"{field}: {value.strip()}")
    return "\n\n".join(parts)


def look_text(profile: Mapping[str, Any]) -> str:
    """Look-for text k_i: the profile's own lookingFor field, tagged."""
    value = profile.get("lookingFor")
    if _nonempty(value):
        return f"lookingFor: {value.strip()}"
    return ""


def seeker_look_text(profile: Mapping[str, Any], search_query: str | None) -> str:
    """Seeker's look-for text k_u: profile lookingFor + this pair's searchQuery."""
    parts: list[str] = []
    base = look_text(profile)
    if base:
        parts.append(base)
    query = (search_query or "").strip()
    if query:
        parts.append(f"Search query: {query}")
    return "\n\n".join(parts)
