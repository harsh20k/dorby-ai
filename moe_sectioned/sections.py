"""Split a seeker's ``lookingFor`` field into its individual asks.

This is the one genuinely new primitive in this experiment. Everything else is a
rearrangement of code that already exists.

**Why splitting on markdown headings and not blank lines.** The existing
``moe_reranker.features._n_sections`` splits on blank lines, which for the first
real pair reports 13 "sections" where the seeker actually wrote 4 asks — it is
counting paragraph blocks. That was harmless when the output was one scalar
among twelve; it is not harmless when it defines the training rows. So this
module splits on ``#``-headings, which is how the field is actually authored, and
falls back to the whole field as a single unnamed section when a seeker wrote no
headings at all.

**Why headings are embedded, not enumerated.** Across the 200 real pairs there
are 994 distinct headings and the eight most common cover only 8.5% of
occurrences. There is no small taxonomy of ask-types to route on, so the heading
text is kept and embedded with its body, and routing has to discover structure
rather than be told it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

_HEADING = re.compile(r"^#{1,6}[ \t]*(.+?)[ \t]*$", re.MULTILINE)


@dataclass(frozen=True)
class Section:
    """One ask, as written by the seeker."""

    #: Position within this seeker's lookingFor, before capping. Kept so a row
    #: can always be traced back to the source field.
    index: int
    #: Heading text, or "" when the seeker wrote no headings.
    heading: str
    body: str

    @property
    def text(self) -> str:
        """What gets embedded: heading and body together.

        The heading carries most of the ask-type signal ("Fundraising",
        "Brand Operators") and the body carries the constraints, so neither is
        useful alone.
        """
        return f"{self.heading}\n{self.body}".strip() if self.heading else self.body


def split_looking_for(
    looking_for: str | None, *, min_chars: int = 40, max_sections: int = 8
) -> list[Section]:
    """Split into asks, drop stubs, cap the tail.

    Capping is not cosmetic. Section counts run from 0 to 116 with a median of 5;
    without a cap a single verbose seeker would contribute more training rows
    than forty ordinary ones, and the model would fit that person.
    """
    if not looking_for or not looking_for.strip():
        return []

    text = looking_for.strip()
    matches = list(_HEADING.finditer(text))

    out: list[Section] = []
    if not matches:
        # No headings — the whole field is one unnamed ask.
        out.append(Section(index=0, heading="", body=text))
    else:
        # Any prose before the first heading belongs to no ask; it is preamble
        # and is dropped rather than silently attached to the first section.
        for i, m in enumerate(matches):
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            out.append(
                Section(index=i, heading=m.group(1).strip(), body=text[start:end].strip())
            )

    kept = [s for s in out if len(s.text) >= min_chars]
    # A seeker whose every section is a stub still needs one row, or the pair
    # vanishes from training entirely. Fall back to the longest.
    if not kept and out:
        kept = [max(out, key=lambda s: len(s.text))]
    return kept[:max_sections]


def sections_for_pair(
    pair: dict[str, Any], *, min_chars: int = 40, max_sections: int = 8
) -> list[Section]:
    seeker = pair.get("userContactFile") or {}
    secs = split_looking_for(
        seeker.get("lookingFor"), min_chars=min_chars, max_sections=max_sections
    )
    if secs:
        return secs
    # No usable lookingFor at all (1 real pair out of 200). The searchQuery is
    # the seeker's ask in that case; if that is empty too the pair yields no
    # rows and is excluded by build_rows(), which reports the count.
    q = (pair.get("searchQuery") or "").strip()
    return [Section(index=0, heading="", body=q)] if q else []


def section_stats(pairs: Sequence[dict[str, Any]], **kw: Any) -> dict[str, float]:
    """Summary used in the run manifest, so a batch's shape is always recorded."""
    counts = [len(sections_for_pair(p, **kw)) for p in pairs]
    n = len(counts) or 1
    return {
        "pairs": len(counts),
        "rows": sum(counts),
        "mean_sections": sum(counts) / n,
        "max_sections": max(counts, default=0),
        "pairs_with_no_sections": sum(1 for c in counts if c == 0),
    }
