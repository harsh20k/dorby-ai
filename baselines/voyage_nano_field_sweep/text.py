"""Parametrized text serialization for the field/query grid sweep.

Deliberately duplicated from ``baselines/voyage_nano_field_selected/text.py``
rather than imported and generalized in place — that package is a separate,
already-published experiment (one fixed field selection); this one takes the
field subset and query flag as parameters instead of hardcoding them, per
the "ML / data-science experiments" isolation rule in CLAUDE.md.
"""

from __future__ import annotations

from typing import Any, Mapping


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _fields_to_text(profile: Mapping[str, Any], fields: tuple[str, ...]) -> str:
    parts: list[str] = []
    for field in fields:
        value = profile.get(field)
        if _nonempty(value):
            parts.append(f"{field}: {value.strip()}")
    return "\n\n".join(parts)


def seeker_to_text(
    user_contact_file: Mapping[str, Any],
    fields: tuple[str, ...],
    search_query: str,
    use_query: bool,
) -> str:
    body = _fields_to_text(user_contact_file, fields)
    if use_query:
        query = (search_query or "").strip()
        if query:
            query_block = f"Search query: {query}"
            return f"{body}\n\n{query_block}" if body else query_block
    return body


def candidate_to_text(match_contact_file: Mapping[str, Any], fields: tuple[str, ...]) -> str:
    return _fields_to_text(match_contact_file, fields)
