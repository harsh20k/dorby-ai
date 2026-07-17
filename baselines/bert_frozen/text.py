"""Build field-tagged text strings from Boardy contact profiles."""

from __future__ import annotations

from typing import Any, Mapping

PROFILE_FIELDS = (
    "positioning",
    "background",
    "lookingFor",
    "notes",
    "locationAvailability",
    "introPreferences",
    "personalPreferences",
    "meetingAndSchedulingPreferences",
)


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def profile_to_text(profile: Mapping[str, Any]) -> str:
    """Concatenate non-empty profile fields with field tags."""
    parts: list[str] = []
    for field in PROFILE_FIELDS:
        value = profile.get(field)
        if _nonempty(value):
            parts.append(f"{field}: {value.strip()}")
    return "\n\n".join(parts)


def seeker_to_text(user_contact_file: Mapping[str, Any], search_query: str) -> str:
    """Seeker text = tagged user profile + search query."""
    body = profile_to_text(user_contact_file)
    query = (search_query or "").strip()
    if query:
        query_block = f"Search query: {query}"
        return f"{body}\n\n{query_block}" if body else query_block
    return body


def candidate_to_text(match_contact_file: Mapping[str, Any]) -> str:
    """Candidate text = tagged match profile only."""
    return profile_to_text(match_contact_file)
