"""Assemble the two populations: real seed pairs and the frozen rrf_003 copy.

Real splits come from ``twotower.data.build_split_bundle`` (the canonical
leakage-safe loader) with ``include_synth=False``, so the promoted `cmsynth*`
pairs from the *old* generator never enter. The synthetic side is read from the
frozen copy under ``artifacts/moe_rrf/../moe_reranker/data/rrf_003/``, which
``moe_reranker/import_rrf.py`` wrote with a source hash — this module never
touches ``artifacts/pairing_rrf/``.

**The evaluation design that makes this experiment worth running.** When training
uses *only* synthetic pairs, every real pair outside the frozen holdout has never
been trained on, so all **131** of them are a legitimate evaluation set. That is
nearly double the 69-pair holdout, which both improves power (SE ~0.070 → ~0.050)
and leaves the one-shot holdout unspent. Arms that train on real pairs cannot use
that trick and fall back to seeker-disjoint cross-validation over the same 131.

Disjointness is asserted, not assumed: every rrf_003 contact id is `cmsynth*`-
prefixed and the intersection with real contact ids is empty (923 vs 1217 ids,
0 overlap at time of writing).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from moe_reranker.import_rrf import load_pairs as load_frozen_rrf
from twotower.data import assert_no_holdout_leak, build_split_bundle

DEFAULT_DATA_DIR = Path("data")
DEFAULT_SPLIT = Path("data/synthetic/seed_split.json")


@dataclass
class Population:
    """A set of labeled pairs with the metadata the experiment needs."""

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


def _to_population(name: str, rows: list[dict[str, Any]], labels: list[str]) -> Population:
    return Population(
        name=name,
        rows=rows,
        y=np.array([1.0 if l == "pos" else 0.0 for l in labels], dtype=np.float32),
        seeker_ids=[r["userContactId"] for r in rows],
    )


def load_real(
    data_dir: Path = DEFAULT_DATA_DIR, split_path: Path = DEFAULT_SPLIT
) -> tuple[Population, Population]:
    """(eval_pool, holdout) — eval_pool is train+train_dev, never trained on by synth arms."""
    bundle = build_split_bundle(data_dir, split_path, include_synth=False)
    assert_no_holdout_leak(bundle, split_path=split_path)

    pool = list(bundle.train) + list(bundle.train_dev)
    eval_pool = _to_population(
        "real_eval_pool", [lp.pair for lp in pool], [lp.label for lp in pool]
    )
    holdout = _to_population(
        "real_holdout",
        [lp.pair for lp in bundle.holdout],
        [lp.label for lp in bundle.holdout],
    )
    return eval_pool, holdout


def load_synth(batch_id: str = "rrf_003") -> Population:
    """The frozen rrf_003 copy. Labels are one LLM judge's opinion, not human outcomes."""
    pairs = load_frozen_rrf(batch_id)
    rows = [p for p in pairs if p.get("label") in ("pos", "neg")]
    return _to_population(batch_id, rows, [p["label"] for p in rows])


def assert_disjoint(a: Population, *others: Population) -> None:
    """No contact id may appear in both populations, on either side of a pair."""

    def ids(p: Population) -> set[str]:
        out: set[str] = set()
        for r in p.rows:
            out.add(r["userContactId"])
            out.add(r["matchContactId"])
        return out

    a_ids = ids(a)
    for o in others:
        overlap = a_ids & ids(o)
        if overlap:
            raise AssertionError(
                f"{a.name} and {o.name} share {len(overlap)} contact ids, e.g. "
                f"{sorted(overlap)[:3]} — the synth-trained evaluation would leak"
            )


def within_seeker_triplets(pop: Population) -> list[tuple[int, int]]:
    """All (positive_row, negative_row) index pairs that share a seeker.

    Training on these instead of on raw pairs cancels the per-seeker base rate by
    construction — the shortcut that scored 0.687 AUC on `rrf_002` with no text at
    all. The real pairs yield 19 of these; rrf_003 yields thousands, which is the
    single concrete thing the synthetic batch unlocks.
    """
    from collections import defaultdict

    by: dict[str, tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
    for i, (sid, label) in enumerate(zip(pop.seeker_ids, pop.y)):
        by[sid][0 if label > 0.5 else 1].append(i)
    out: list[tuple[int, int]] = []
    for pos_idx, neg_idx in by.values():
        for p in pos_idx:
            for n in neg_idx:
                out.append((p, n))
    return out


def seeker_disjoint_folds(
    seeker_ids: Sequence[str], k: int, seed: int
) -> list[np.ndarray]:
    """Whole seekers assigned to folds, so no seeker spans train and validation."""
    rng = np.random.default_rng(seed)
    uniq = sorted(set(seeker_ids))
    rng.shuffle(uniq)
    fold_of = {s: i % k for i, s in enumerate(uniq)}
    assign = np.array([fold_of[s] for s in seeker_ids])
    return [np.where(assign == f)[0] for f in range(k)]
