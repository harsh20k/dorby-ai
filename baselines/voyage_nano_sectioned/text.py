"""Per-section candidate text variants, splitting lookingFor on its own blank-line breaks.

Boardy lookingFor fields are typically already written as several independently
labeled asks (e.g. "### Pilot Customers" / "### Fundraising" / "### VPs of R&D"),
each its own paragraph. Embedding the whole field as one string blurs those asks
into a single averaged vector; a seeker asking about only one of them then scores
worse than they should. This module keeps the split literal: paragraph breaks
that already exist in the data, no header parsing or inference.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from baselines.bert_frozen.text import _nonempty, profile_to_text, seeker_to_text

_BLANK_LINE_RE = re.compile(r"\n\s*\n")


def split_looking_for_sections(looking_for: str) -> list[str]:
    """Split a lookingFor field on blank-line paragraph breaks."""
    parts = [p.strip() for p in _BLANK_LINE_RE.split(looking_for.strip())]
    return [p for p in parts if p]


def candidate_to_sectioned_texts(match_contact_file: Mapping[str, Any]) -> list[str]:
    """Full candidate profile text, once per lookingFor section (other fields unchanged).

    Falls back to a single whole-profile text (identical to
    ``bert_frozen.text.candidate_to_text``) when lookingFor is empty or has
    only one section, so single-ask candidates are scored exactly as today.
    """
    looking_for = match_contact_file.get("lookingFor")
    if not _nonempty(looking_for):
        return [profile_to_text(match_contact_file)]

    sections = split_looking_for_sections(looking_for)
    if len(sections) <= 1:
        return [profile_to_text(match_contact_file)]

    texts: list[str] = []
    for section in sections:
        variant = dict(match_contact_file)
        variant["lookingFor"] = section
        texts.append(profile_to_text(variant))
    return texts


def seeker_to_sectioned_texts(
    user_contact_file: Mapping[str, Any], search_query: str
) -> list[str]:
    """Mirror of ``candidate_to_sectioned_texts`` for the seeker side.

    Splits the seeker's own lookingFor field into sections and builds one
    seeker text per section (profile + searchQuery unchanged each time,
    lookingFor swapped to one section). Falls back to a single whole-profile
    seeker text (identical to ``bert_frozen.text.seeker_to_text``) when
    lookingFor is empty or has only one section.
    """
    looking_for = user_contact_file.get("lookingFor")
    if not _nonempty(looking_for):
        return [seeker_to_text(user_contact_file, search_query)]

    sections = split_looking_for_sections(looking_for)
    if len(sections) <= 1:
        return [seeker_to_text(user_contact_file, search_query)]

    texts: list[str] = []
    for section in sections:
        variant = dict(user_contact_file)
        variant["lookingFor"] = section
        texts.append(seeker_to_text(variant, search_query))
    return texts
