"""Load the frozen ask/offer rows (written by import_rows.py) and carve a
seeker-disjoint dev split.

Same seeker-disjoint-by-id convention as `twotower.data.carve_train_dev` /
`twotower_voyage_gemini_ctrl.data.carve_dev`, reimplemented here to keep this
package import-free of both (same reasoning as
`twotower_voyage_gemini_ctrl/data.py`'s own docstring: these rows never touch
a real user, so there is no holdout-leak surface against the real
seed_split — the real 69-pair holdout is only ever touched by
twotower_ask_offer.eval at the very end).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class AskOfferRow:
    query_key: str
    seeker_id: str
    positive_id: str
    negative_ids: tuple[str, ...]
    search_query: str
    seeker_profile: Mapping[str, Any]
    positive_profile: Mapping[str, Any]
    negative_profiles: tuple[Mapping[str, Any], ...]


def load_rows(path: Path) -> tuple[list[AskOfferRow], dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = [
        AskOfferRow(
            query_key=r["query_key"],
            seeker_id=r["seeker_id"],
            positive_id=r["positive_id"],
            negative_ids=tuple(r["negative_ids"]),
            search_query=r["search_query"],
            seeker_profile=r["seeker_profile"],
            positive_profile=r["positive_profile"],
            negative_profiles=tuple(r["negative_profiles"]),
        )
        for r in payload["rows"]
    ]
    return rows, payload["provenance"]


def carve_dev(
    rows: list[AskOfferRow],
    *,
    seeker_fraction: float = 0.1,
    min_rows: int = 20,
    seed: int = 42,
) -> tuple[list[AskOfferRow], list[AskOfferRow]]:
    if not rows:
        return [], []

    by_seeker: dict[str, list[AskOfferRow]] = {}
    for r in rows:
        by_seeker.setdefault(r.seeker_id, []).append(r)

    seekers = sorted(by_seeker)
    rng = random.Random(seed)
    rng.shuffle(seekers)

    n_dev_seekers = max(1, int(round(len(seekers) * seeker_fraction)))
    dev_seekers: set[str] = set()
    for sid in seekers:
        if len(dev_seekers) >= n_dev_seekers:
            n_rows = sum(len(by_seeker[s]) for s in dev_seekers)
            if n_rows >= min_rows:
                break
        if len(dev_seekers) >= max(1, len(seekers) // 2):
            break
        dev_seekers.add(sid)

    train = [r for r in rows if r.seeker_id not in dev_seekers]
    dev = [r for r in rows if r.seeker_id in dev_seekers]
    return train, dev
