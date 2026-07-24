"""Candidate text packing with exactly one profile field ablated.

Seeker-side packing is unchanged — re-export ``seeker_to_text`` from
``baselines.bert_frozen.text``. Mirrors ``baselines/text_no_query.py``'s
shape (one small module, ablates one thing, reuses everything else) but
ablates a candidate-side profile field instead of the seeker-side query.
"""

from __future__ import annotations

from typing import Any, Mapping

from baselines.bert_frozen.text import PROFILE_FIELDS, seeker_to_text

__all__ = ["PROFILE_FIELDS", "candidate_to_text_ablate_field", "seeker_to_text"]


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def candidate_to_text_ablate_field(match_contact_file: Mapping[str, Any], ablate_field: str) -> str:
    """Candidate text = tagged match profile, with `ablate_field` dropped entirely."""
    if ablate_field not in PROFILE_FIELDS:
        raise ValueError(f"unknown field {ablate_field!r}, expected one of {PROFILE_FIELDS}")
    parts: list[str] = []
    for field in PROFILE_FIELDS:
        if field == ablate_field:
            continue
        value = match_contact_file.get(field)
        if _nonempty(value):
            parts.append(f"{field}: {value.strip()}")
    return "\n\n".join(parts)
