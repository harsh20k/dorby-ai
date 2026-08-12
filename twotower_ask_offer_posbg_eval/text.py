"""Offer text = positioning + background only.

Copied from baselines/reciprocal_lambda_grid/text.py (isolation rule).
Ask-side look_text / seeker_look_text match reciprocal_static.
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
