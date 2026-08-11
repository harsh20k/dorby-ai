"""Multi-negative row loading + seeker-disjoint dev carve.

Verbatim copy of ``twotower_top1_optimised/data.py`` (not imported from it,
per the isolation rule) — this loader is generic over whatever text sits in
``anchor``/``positive``/``negatives``, so it needs no changes for the
trimmed-field row file this package trains on. Pinned byte-for-byte by
``tests/test_field_pos_bg.py``.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MultiNegRow:
    query_key: str
    seeker_id: str
    positive_id: str
    negative_ids: tuple[str, ...]
    anchor: str
    positive: str
    negatives: tuple[str, ...]
    n_unique_negatives: int
    padded_count: int


def load_multineg_rows(path: Path) -> tuple[list[MultiNegRow], dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = [
        MultiNegRow(
            query_key=r["query_key"],
            seeker_id=r["seeker_id"],
            positive_id=r["positive_id"],
            negative_ids=tuple(r["negative_ids"]),
            anchor=r["anchor"],
            positive=r["positive"],
            negatives=tuple(r["negatives"]),
            n_unique_negatives=r["n_unique_negatives"],
            padded_count=r["padded_count"],
        )
        for r in payload["rows"]
    ]
    return rows, payload["summary"]


def carve_dev(
    rows: list[MultiNegRow],
    *,
    seeker_fraction: float = 0.1,
    min_rows: int = 20,
    seed: int = 42,
) -> tuple[list[MultiNegRow], list[MultiNegRow]]:
    """Seeker-disjoint train / dev split (same unit-of-disjointness convention
    as twotower.data.carve_train_dev / twotower_rrf_triplet.data.carve_dev,
    reimplemented here to keep this package import-free of both)."""
    if not rows:
        return [], []

    by_seeker: dict[str, list[MultiNegRow]] = {}
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


def rows_to_hf_dict(rows: list[MultiNegRow], *, negatives_per_anchor: int) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {
        "anchor": [r.anchor for r in rows],
        "positive": [r.positive for r in rows],
    }
    for i in range(negatives_per_anchor):
        out[f"negative_{i + 1}"] = [r.negatives[i] for r in rows]
    return out
