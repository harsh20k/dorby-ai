"""Seeker-side text = exactly two identity-carrying fields, no query, no other field.

Delegates field tagging to ``baselines.bert_frozen.text`` (same ``"field:
value"`` format, same ``\\n\\n`` join, same non-empty check) so a two-field
string here is a strict substring/subset of what ``profile_to_text`` would
produce for the full profile — nothing is reformatted, only omitted.
"""

from __future__ import annotations

from typing import Any, Mapping

from baselines.bert_frozen.text import candidate_to_text

__all__ = [
    "background_lookingfor",
    "candidate_to_text",
    "pos_background",
    "pos_lookingfor",
]


def _fields_to_text(profile: Mapping[str, Any], fields: tuple[str, ...]) -> str:
    # Same non-empty check as baselines.bert_frozen.text.profile_to_text,
    # reimplemented rather than importing the module-private helper.
    parts: list[str] = []
    for field in fields:
        value = profile.get(field)
        if isinstance(value, str) and value.strip():
            parts.append(f"{field}: {value.strip()}")
    return "\n\n".join(parts)


def pos_background(user_contact_file: Mapping[str, Any], search_query: str = "") -> str:
    """positioning + background. ``search_query`` accepted and ignored (uniform arm signature)."""
    return _fields_to_text(user_contact_file, ("positioning", "background"))


def pos_lookingfor(user_contact_file: Mapping[str, Any], search_query: str = "") -> str:
    """positioning + lookingFor."""
    return _fields_to_text(user_contact_file, ("positioning", "lookingFor"))


def background_lookingfor(user_contact_file: Mapping[str, Any], search_query: str = "") -> str:
    """background + lookingFor."""
    return _fields_to_text(user_contact_file, ("background", "lookingFor"))
