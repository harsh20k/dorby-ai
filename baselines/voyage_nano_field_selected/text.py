"""Field-selected text serialization, matching the focused LLM-judge prompt.

Deliberately duplicated from
``baselines/llm_judge_with_pos_look_pos_back_look/prompt.py``'s field lists
rather than imported — that module lives in an LLM-prompt package, this one
in an embedding-baseline package, and the two may need to diverge in what
"field selection" means for a prompt vs. a bi-encoder (e.g. no JSON response
contract to strip out here). Small enough that duplication costs nothing and
keeps each package's own history clean.
"""

from __future__ import annotations

from typing import Any, Mapping

SEEKER_FIELDS = ("positioning", "lookingFor")
CANDIDATE_FIELDS = ("positioning", "background", "lookingFor")


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _fields_to_text(profile: Mapping[str, Any], fields: tuple[str, ...]) -> str:
    parts: list[str] = []
    for field in fields:
        value = profile.get(field)
        if _nonempty(value):
            parts.append(f"{field}: {value.strip()}")
    return "\n\n".join(parts)


def seeker_to_text(user_contact_file: Mapping[str, Any], search_query: str) -> str:
    """Seeker text = positioning + lookingFor + searchQuery (not the full profile)."""
    body = _fields_to_text(user_contact_file, SEEKER_FIELDS)
    query = (search_query or "").strip()
    if query:
        query_block = f"Search query: {query}"
        return f"{body}\n\n{query_block}" if body else query_block
    return body


def candidate_to_text(match_contact_file: Mapping[str, Any]) -> str:
    """Candidate text = positioning + background + lookingFor only."""
    return _fields_to_text(match_contact_file, CANDIDATE_FIELDS)
