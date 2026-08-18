"""Seeker = query only; candidate = background + lookingFor.

Copied from query_weighted.text.query_only and
field_pairs_sweep.text.background_lookingfor (isolation rule). Drift pinned
by tests/test_bdata_queryonly_back_look.py.
"""

from __future__ import annotations

from typing import Any, Mapping

from baselines.bert_frozen.text import profile_to_text

QUERY_PREFIX = "Search query: "


def query_block(search_query: str) -> str:
    q = (search_query or "").strip()
    return f"{QUERY_PREFIX}{q}" if q else ""


def query_only(user_contact_file: Mapping[str, Any], search_query: str) -> str:
    """Ask with no biography. Empty query falls back to the full profile."""
    block = query_block(search_query)
    return block or profile_to_text(user_contact_file)


def background_lookingfor(user_contact_file: Mapping[str, Any], search_query: str = "") -> str:
    """background + lookingFor. search_query accepted and ignored."""
    del search_query
    parts: list[str] = []
    for field in ("background", "lookingFor"):
        value = user_contact_file.get(field)
        if isinstance(value, str) and value.strip():
            parts.append(f"{field}: {value.strip()}")
    return "\n\n".join(parts)
