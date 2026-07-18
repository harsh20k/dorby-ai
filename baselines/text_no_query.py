"""Seeker text packing with searchQuery ablated (profile-only).

Candidate packing is unchanged — re-export ``candidate_to_text`` from
``baselines.bert_frozen.text``. Does not alter the with-query packing used by
the original baselines.
"""

from __future__ import annotations

from typing import Any, Mapping

from baselines.bert_frozen.text import candidate_to_text, profile_to_text

__all__ = ["seeker_to_text", "candidate_to_text", "profile_to_text"]


def seeker_to_text(user_contact_file: Mapping[str, Any]) -> str:
    """Seeker text = tagged user profile only (no Search query block)."""
    return profile_to_text(user_contact_file)
