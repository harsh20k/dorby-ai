"""Seeker sectioning: turn one profile into N+1 embedding texts.

A seeker with two ``lookingFor`` sections yields three vectors — one for the
complete profile, plus one per section carrying that section and every other
profile field but dropping its sibling sections.

The reason for keeping both kinds is measured, not aesthetic. On the 69-pair
holdout, seeker-sectioning lifted pair AUC 0.579 → 0.596 and top-1 retrieval
27.6% → 34.5%, but cost Recall@10 (0.759 → 0.690), and no aggregation softening
recovered it — max, top-2 mean and softmax all landed on the same 0.690. Keeping
the whole-profile vector alongside the sharp per-section ones is what buys the
precision without giving up the breadth. See ``docs/lookingfor-sectioning-findings.md``.

The split itself is literal — blank-line paragraph breaks already present in the
data, shared with ``baselines/voyage_nano_sectioned`` so the two cannot drift.
No header parsing, no LLM, no inference.

**The search query is never part of any embedding text here.** Candidates are
embedded as whole profiles only, because splitting the candidate side measured
*worse* on every metric (0.568 vs 0.579 baseline) — a candidate means all of
their asks, so there is nothing to un-blur on that side.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from baselines.bert_frozen.text import _nonempty, profile_to_text
from baselines.voyage_nano_sectioned.text import split_looking_for_sections

WHOLE = -1  # section_index sentinel for the complete-profile vector


@dataclass(frozen=True)
class SeekerVector:
    """One embedding text for a seeker, plus what it was built from."""

    contact_id: str
    section_index: int  # WHOLE for the complete profile, else 0-based
    section_text: str | None  # None for the whole-profile vector
    text: str

    @property
    def key(self) -> str:
        """Stable id used as the vector-store primary key."""
        suffix = "whole" if self.section_index == WHOLE else f"s{self.section_index}"
        return f"{self.contact_id}::{suffix}"

    @property
    def is_whole(self) -> bool:
        return self.section_index == WHOLE


def looking_for_sections(profile: Mapping[str, Any]) -> list[str]:
    """Sections of this profile's ``lookingFor``; [] when the field is empty."""
    looking_for = profile.get("lookingFor")
    if not _nonempty(looking_for):
        return []
    return split_looking_for_sections(looking_for)


def seeker_vectors(contact_id: str, profile: Mapping[str, Any]) -> list[SeekerVector]:
    """Build the N+1 embedding texts for one seeker.

    Returns just the whole-profile vector when ``lookingFor`` has fewer than two
    sections — with one section the per-section text would be byte-identical to
    the whole, and paying to store the duplicate buys nothing.
    """
    whole = SeekerVector(
        contact_id=contact_id,
        section_index=WHOLE,
        section_text=None,
        text=profile_to_text(profile),
    )
    sections = looking_for_sections(profile)
    if len(sections) <= 1:
        return [whole]

    out = [whole]
    for i, section in enumerate(sections):
        variant = dict(profile)
        variant["lookingFor"] = section
        out.append(
            SeekerVector(
                contact_id=contact_id,
                section_index=i,
                section_text=section,
                text=profile_to_text(variant),
            )
        )
    return out


def candidate_text(profile: Mapping[str, Any]) -> str:
    """Candidate embedding text — the whole profile, never sectioned, never a query."""
    return profile_to_text(profile)


@dataclass(frozen=True)
class QueryTarget:
    """One (seeker, section) pair that needs a ``searchQuery`` written for it."""

    contact_id: str
    section_index: int
    section_text: str

    @property
    def key(self) -> str:
        return f"{self.contact_id}::q{self.section_index}"


def query_targets(contact_id: str, profile: Mapping[str, Any]) -> list[QueryTarget]:
    """One query per ``lookingFor`` section.

    A profile whose ``lookingFor`` does not split still gets a single query,
    written against the whole field — otherwise single-ask seekers would drop
    out of the batch entirely.
    """
    sections = looking_for_sections(profile)
    if not sections:
        return []
    return [
        QueryTarget(contact_id=contact_id, section_index=i, section_text=section)
        for i, section in enumerate(sections)
    ]
