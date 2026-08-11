"""Seeker/candidate text composition for the Voyage-4-large + Gemini pairing batch.

Field choice is dictated by the query-field R@10 sweep, not aesthetics:

* Seeker embedding text = ``positioning`` + this query's generated ``searchQuery``
  only. One embedding vector per query (one per ``lookingFor`` section), not one
  per seeker — since the query text differs per section, so does the vector.
* Candidate embedding text = ``positioning`` + ``background`` + ``lookingFor``.

Distinct from ``synth_pipeline.pairing_rrf_qwen_judge.sections``, which embeds the
seeker's *whole profile* per section and never includes the query in the
embedding text at all. Query targets/text here are not regenerated — this batch
reuses ``queries.json`` from ``rrf_qwen_full_001`` verbatim (see ``run.py``), so
only the section-splitting logic (identifying which query goes with which
profile) is duplicated, not the query generation itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from baselines.bert_frozen.text import _nonempty
from baselines.voyage_nano_sectioned.text import split_looking_for_sections

SEEKER_FIELDS = ("positioning",)
CANDIDATE_FIELDS = ("positioning", "background", "lookingFor")


def _fields_to_text(profile: Mapping[str, Any], fields: tuple[str, ...]) -> str:
    parts: list[str] = []
    for field_name in fields:
        value = profile.get(field_name)
        if _nonempty(value):
            parts.append(f"{field_name}: {value.strip()}")
    return "\n\n".join(parts)


@dataclass(frozen=True)
class QueryTarget:
    """One (seeker, section) pair — matches a key already present in the reused queries.json."""

    contact_id: str
    section_index: int
    section_text: str

    @property
    def key(self) -> str:
        return f"{self.contact_id}::q{self.section_index}"


def looking_for_sections(profile: Mapping[str, Any]) -> list[str]:
    looking_for = profile.get("lookingFor")
    if not _nonempty(looking_for):
        return []
    return split_looking_for_sections(looking_for)


def query_targets(contact_id: str, profile: Mapping[str, Any]) -> list[QueryTarget]:
    """Same split ``query_gen`` used to write queries.json — needed to know which
    reused query keys belong to this profile, not to generate anything new."""
    sections = looking_for_sections(profile)
    if not sections:
        return []
    return [
        QueryTarget(contact_id=contact_id, section_index=i, section_text=s)
        for i, s in enumerate(sections)
    ]


def seeker_query_text(profile: Mapping[str, Any], search_query: str) -> str:
    """Seeker embedding text = positioning + this section's searchQuery."""
    body = _fields_to_text(profile, SEEKER_FIELDS)
    query = (search_query or "").strip()
    if query:
        block = f"searchQuery: {query}"
        return f"{body}\n\n{block}" if body else block
    return body


def candidate_text(profile: Mapping[str, Any]) -> str:
    """Candidate embedding text = positioning + background + lookingFor."""
    return _fields_to_text(profile, CANDIDATE_FIELDS)
