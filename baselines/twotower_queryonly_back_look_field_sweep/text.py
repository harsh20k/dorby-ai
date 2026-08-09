"""Parametrized text serialization for the field/query grid sweep.

Byte-identical duplicate of
``baselines/twotower_top1_ctrl_field_sweep/text.py`` (same field-subset x
query-flag parametrization), copied rather than imported — this package
re-runs that same 105-combo sweep design against a different encoder (the
fine-tuned ``queryonly_back_look_001`` LoRA adapter), per the "ML /
data-science experiments" isolation rule in CLAUDE.md: a new encoder is a
new experiment, not a new mode of an existing one.
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
