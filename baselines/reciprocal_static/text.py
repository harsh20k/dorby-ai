"""Look-for vs. background text split for the static reciprocal two-tower experiment.

Ga Wu's "Dynamic Reciprocal User Matching with Fast Weight Programmers" (Boardy
AI, 2026-06-07) defines two per-user embeddings: k_i = E_look(look-for text)
and v_i = E_bg(background text) (its eqs. 1-2). This module builds those two
text views from a Boardy contact profile.

``PROFILE_FIELDS`` is reused read-only from ``baselines.bert_frozen.text``
(single source of truth for the field list); this module only partitions it,
never edits that file.
"""

from __future__ import annotations

from typing import Any, Mapping

from baselines.bert_frozen.text import PROFILE_FIELDS

BG_FIELDS = tuple(field for field in PROFILE_FIELDS if field != "lookingFor")


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def bg_text(profile: Mapping[str, Any]) -> str:
    """Background text v_i: every profile field except lookingFor, tagged."""
    parts: list[str] = []
    for field in BG_FIELDS:
        value = profile.get(field)
        if _nonempty(value):
            parts.append(f"{field}: {value.strip()}")
    return "\n\n".join(parts)


def look_text(profile: Mapping[str, Any]) -> str:
    """Look-for text k_i: the profile's own lookingFor field, tagged.

    Used for a user's reciprocal-side embedding (the paper's k_i in
    s_reciprocal = k_i^T v_u) — their standing stated preference, with no
    per-interaction search query, since a candidate has no query in the
    direction being scored.
    """
    value = profile.get("lookingFor")
    if _nonempty(value):
        return f"lookingFor: {value.strip()}"
    return ""


def seeker_look_text(profile: Mapping[str, Any], search_query: str | None) -> str:
    """Seeker's look-for text k_u: profile lookingFor + this pair's searchQuery.

    The paper has no separate "query" concept (eq. 1 defines k_i purely from
    the standing profile text); searchQuery is this repo's per-interaction
    demand signal and CLAUDE.md's query-ablation finding says it is
    load-bearing, not redundant with the profile. It belongs on the k_u
    (look-for/demand) side of the split, never on v_u (background/supply).
    """
    parts: list[str] = []
    base = look_text(profile)
    if base:
        parts.append(base)
    query = (search_query or "").strip()
    if query:
        parts.append(f"Search query: {query}")
    return "\n\n".join(parts)
