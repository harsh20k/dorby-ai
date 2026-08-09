"""Combo enumeration for the field/query grid sweep.

Byte-identical duplicate of ``baselines/twotower_top1_ctrl_field_sweep/fields.py``
(same grid design, different encoder — see this package's ``__init__.py``),
copied rather than imported per the isolation rule in CLAUDE.md.

7 non-empty subsets of (positioning, background, lookingFor) per side x 7 x
2 query settings = 98 combinations, plus a "query-only" seeker (no profile
fields at all, searchQuery is the entire seeker text) x the same 7 candidate
subsets = 7 more, for 105 total. Field order within a subset always follows
FIELDS, so e.g. "background+positioning" and "positioning+background" never
appear as separate combos.

The query-only seeker is forced to use_query=True — an empty seeker with the
query also excluded would be a fully empty seeker text (no signal at all),
which isn't a meaningful combo, so that pairing is skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

FIELDS: tuple[str, ...] = ("positioning", "background", "lookingFor")

_SHORT = {"positioning": "pos", "background": "back", "lookingFor": "look"}


def _non_empty_subsets(fields: tuple[str, ...]) -> list[tuple[str, ...]]:
    subsets: list[tuple[str, ...]] = []
    for r in range(1, len(fields) + 1):
        subsets.extend(combinations(fields, r))
    return subsets


FIELD_SUBSETS: list[tuple[str, ...]] = _non_empty_subsets(FIELDS)  # 7 subsets

# Seeker side additionally allows the empty subset (query-only seeker);
# candidate side never does — "normal combinations" only, per design.
SEEKER_SUBSETS: list[tuple[str, ...]] = [()] + FIELD_SUBSETS  # 8 subsets


def subset_id(fields: tuple[str, ...]) -> str:
    return "_".join(_SHORT[f] for f in fields) or "none"


@dataclass(frozen=True)
class Combo:
    combo_id: str
    seeker_fields: tuple[str, ...]
    candidate_fields: tuple[str, ...]
    use_query: bool


def all_combos() -> list[Combo]:
    combos: list[Combo] = []
    for seeker_fields in SEEKER_SUBSETS:
        for candidate_fields in FIELD_SUBSETS:
            query_settings = (True,) if not seeker_fields else (True, False)
            for use_query in query_settings:
                combo_id = (
                    f"seeker={subset_id(seeker_fields)}"
                    f"__cand={subset_id(candidate_fields)}"
                    f"__query={'yes' if use_query else 'no'}"
                )
                combos.append(
                    Combo(
                        combo_id=combo_id,
                        seeker_fields=seeker_fields,
                        candidate_fields=candidate_fields,
                        use_query=use_query,
                    )
                )
    return combos


if __name__ == "__main__":
    combos = all_combos()
    print(
        f"{len(FIELD_SUBSETS)} non-empty field subsets x {len(FIELD_SUBSETS)} x 2 query settings "
        f"= 98, plus query-only-seeker x {len(FIELD_SUBSETS)} candidates = 7 more "
        f"= {len(combos)} combos"
    )
    assert len(combos) == 105
