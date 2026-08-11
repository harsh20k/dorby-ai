"""Seeker-side text variants. The candidate side is never touched.

``concat_baseline`` and ``profile_only`` delegate to ``baselines.bert_frozen.text``
rather than reimplementing it — they must be byte-identical to what the published
baselines encode, or the comparison has no anchor. The remaining builders are new
and exist only here.

The ``Search query: `` prefix is duplicated from ``seeker_to_text``'s internals
because that function offers no way to compose the block separately. The test
suite asserts the duplicate still agrees with the original by round-tripping a
real pair through both.
"""

from __future__ import annotations

from typing import Any, Mapping

from baselines.bert_frozen.text import (
    candidate_to_text,
    profile_to_text,
    seeker_to_text,
)

__all__ = [
    "QUERY_PREFIX",
    "candidate_to_text",
    "concat_baseline",
    "profile_only",
    "query_block",
    "query_first",
    "query_only",
    "query_repeated_front",
]

# Must match baselines/bert_frozen/text.py::seeker_to_text.
QUERY_PREFIX = "Search query: "
_SEP = "\n\n"


def query_block(search_query: str) -> str:
    """``Search query: …``, or empty when the pair has no query."""
    q = (search_query or "").strip()
    return f"{QUERY_PREFIX}{q}" if q else ""


def profile_only(user_contact_file: Mapping[str, Any], search_query: str = "") -> str:
    """Seeker profile with the query removed entirely (the ``no_query`` ablation).

    ``search_query`` is accepted and ignored so every builder shares one
    signature and the arm table can call them uniformly.
    """
    return profile_to_text(user_contact_file)


def concat_baseline(user_contact_file: Mapping[str, Any], search_query: str) -> str:
    """Exactly what every published baseline encodes: profile, then query."""
    return seeker_to_text(user_contact_file, search_query)


def query_only(user_contact_file: Mapping[str, Any], search_query: str) -> str:
    """The ask with no biography at all — the α=1 end of the weighting sweep.

    Falls back to the profile when a pair has an empty query, so this arm never
    encodes an empty string (which would produce a meaningless vector rather
    than a fair measurement).
    """
    block = query_block(search_query)
    return block or profile_to_text(user_contact_file)


def query_first(user_contact_file: Mapping[str, Any], search_query: str) -> str:
    """Same tokens as the baseline, query moved to the front.

    Isolates *position* from *quantity*: identical content to
    ``concat_baseline``, so any difference is attention/position, not weight.
    """
    body = profile_to_text(user_contact_file)
    block = query_block(search_query)
    if not block:
        return body
    return f"{block}{_SEP}{body}" if body else block


def query_repeated_front(
    user_contact_file: Mapping[str, Any],
    search_query: str,
    repeats: int = 3,
) -> str:
    """Query repeated ``repeats`` times at the front, then the profile.

    Crude but direct: it raises the query's share of the token budget without
    changing the model. At repeats=1 this is exactly ``query_first``, which is
    what makes the repeat count a clean one-dimensional knob.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    body = profile_to_text(user_contact_file)
    block = query_block(search_query)
    if not block:
        return body
    head = _SEP.join([block] * repeats)
    return f"{head}{_SEP}{body}" if body else head
