"""Row loading for the field-gate experiment. Deliberate near-copy of
twotower_query_only/data.py's MultiNegRow shape, adapted for a `pieces` dict
on the anchor side instead of one flat string."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FieldRow:
    query_key: str
    seeker_id: str
    positive_id: str
    negative_ids: tuple[str, ...]
    pieces: dict[str, str]  # {"query": ..., "lookingFor": ..., "positioning": ...}
    positive: str
    negatives: tuple[str, ...]


def load_field_rows(path: Path) -> tuple[list[FieldRow], dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = [
        FieldRow(
            query_key=r["query_key"],
            seeker_id=r["seeker_id"],
            positive_id=r["positive_id"],
            negative_ids=tuple(r["negative_ids"]),
            pieces=r["pieces"],
            positive=r["positive"],
            negatives=tuple(r["negatives"]),
        )
        for r in payload["rows"]
    ]
    return rows, payload["summary"]


def carve_dev(
    rows: list[FieldRow],
    *,
    seeker_fraction: float = 0.1,
    min_rows: int = 20,
    seed: int = 42,
) -> tuple[list[FieldRow], list[FieldRow]]:
    """Seeker-disjoint train/dev split — same convention as every sibling package."""
    if not rows:
        return [], []
    by_seeker: dict[str, list[FieldRow]] = {}
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
