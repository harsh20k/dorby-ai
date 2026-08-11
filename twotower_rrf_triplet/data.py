"""Triplet loading + seeker-disjoint dev carve for the rrf_003 experiment.

Fully separate from twotower/data.py's SplitBundle machinery: these triplets
come from LLM-judge-labeled synthetic profiles (rrf_003), never touch a real
user, and therefore have no holdout-leak surface against the real seed_split —
the real 69-pair holdout stays reserved for the one number that matters
(final evaluation, done via twotower.eval, not this module).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Triplet:
    query_key: str
    seeker_id: str
    positive_id: str
    negative_id: str
    anchor: str
    positive: str
    negative: str


def load_triplets(path: Path) -> tuple[list[Triplet], dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = [Triplet(**row) for row in payload["triplets"]]
    return rows, payload["summary"]


def carve_dev(
    triplets: list[Triplet],
    *,
    seeker_fraction: float = 0.1,
    min_triplets: int = 20,
    seed: int = 42,
) -> tuple[list[Triplet], list[Triplet]]:
    """Seeker-disjoint train / dev split (same unit-of-disjointness convention
    as twotower.data.carve_train_dev, reimplemented here to avoid any import
    coupling to the real-data split path)."""
    if not triplets:
        return [], []

    by_seeker: dict[str, list[Triplet]] = {}
    for t in triplets:
        by_seeker.setdefault(t.seeker_id, []).append(t)

    seekers = sorted(by_seeker)
    rng = random.Random(seed)
    rng.shuffle(seekers)

    n_dev_seekers = max(1, int(round(len(seekers) * seeker_fraction)))
    dev_seekers: set[str] = set()
    for sid in seekers:
        if len(dev_seekers) >= n_dev_seekers:
            n_rows = sum(len(by_seeker[s]) for s in dev_seekers)
            if n_rows >= min_triplets:
                break
        if len(dev_seekers) >= max(1, len(seekers) // 2):
            break
        dev_seekers.add(sid)

    train = [t for t in triplets if t.seeker_id not in dev_seekers]
    dev = [t for t in triplets if t.seeker_id in dev_seekers]
    return train, dev


def triplets_to_hf_dict(rows: list[Triplet]) -> dict[str, list[str]]:
    return {
        "anchor": [t.anchor for t in rows],
        "positive": [t.positive for t in rows],
        "negative": [t.negative for t in rows],
    }
