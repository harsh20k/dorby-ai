"""Load real pairs, explode into section rows, and carve seeker-disjoint folds.

Splits come from ``twotower.data.build_split_bundle`` — the canonical
leakage-safe loader — with ``include_synth=False``, so the quarantined
`batch_500_001` pairs never enter (see
``data/archive/batch_500_001_quarantined/README.md``).

**Folds are by seeker, never by row or by pair.** A seeker contributes several
sections and often several pairs; splitting anywhere below the seeker would put
the same person's asks on both sides of the split, and an earlier batch showed
seeker identity alone predicting the label at 0.687 AUC. Whole seekers move
together or the number is meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from twotower.data import assert_no_holdout_leak, build_split_bundle


@dataclass
class Population:
    name: str
    rows: list[dict[str, Any]]
    y: np.ndarray
    seeker_ids: list[str]

    def __len__(self) -> int:
        return len(self.rows)

    def summary(self) -> str:
        pos = int(self.y.sum())
        return (
            f"{self.name}: {len(self.rows)} pairs ({pos} pos / {len(self.rows) - pos} neg), "
            f"{len(set(self.seeker_ids))} seekers"
        )


def _to_population(name: str, pairs: list[dict[str, Any]], labels: list[str]) -> Population:
    return Population(
        name=name,
        rows=pairs,
        y=np.array([1.0 if l == "pos" else 0.0 for l in labels], dtype=np.float32),
        seeker_ids=[p["userContactId"] for p in pairs],
    )


def load_real(
    data_dir: Path = Path("data"),
    split_path: Path = Path("data/synthetic/seed_split.json"),
) -> tuple[Population, Population]:
    """(train_pool, holdout).

    ``train_pool`` is the 131 real pairs outside the frozen holdout; ``holdout``
    is the frozen 69 and must be scored at most once, at the end, and only if the
    cross-validated result has already cleared the bar.
    """
    bundle = build_split_bundle(data_dir, split_path, include_synth=False)
    assert_no_holdout_leak(bundle, split_path=split_path)

    pool = list(bundle.train) + list(bundle.train_dev)
    return (
        _to_population("real_train_pool", [lp.pair for lp in pool], [lp.label for lp in pool]),
        _to_population(
            "real_holdout",
            [lp.pair for lp in bundle.holdout],
            [lp.label for lp in bundle.holdout],
        ),
    )


def seeker_disjoint_folds(
    seeker_ids: Sequence[str], k: int, seed: int
) -> list[np.ndarray]:
    """Whole seekers assigned to folds; returns pair indices per fold."""
    rng = np.random.default_rng(seed)
    uniq = sorted(set(seeker_ids))
    rng.shuffle(uniq)
    fold_of = {s: i % k for i, s in enumerate(uniq)}
    assign = np.array([fold_of[s] for s in seeker_ids])
    return [np.where(assign == f)[0] for f in range(k)]


def rows_for_pairs(groups: list[np.ndarray], pair_idx: np.ndarray) -> np.ndarray:
    """Row indices belonging to the given pairs — used to slice a fold's rows."""
    if len(pair_idx) == 0:
        return np.array([], dtype=np.int64)
    return np.concatenate([groups[p] for p in pair_idx])


def regroup(groups: list[np.ndarray], pair_idx: np.ndarray) -> list[np.ndarray]:
    """Re-index a subset of pairs' groups onto a compacted row array.

    After slicing rows for a fold, the original row indices no longer address the
    sliced matrices, so the per-pair groups have to be renumbered to match.
    """
    out, cursor = [], 0
    for p in pair_idx:
        n = len(groups[p])
        out.append(np.arange(cursor, cursor + n, dtype=np.int64))
        cursor += n
    return out
