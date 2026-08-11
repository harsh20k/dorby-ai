"""Load the 200 real pairs as ``LabeledPair``s, verified against the frozen manifest.

``twotower.data.build_split_bundle`` deliberately cannot express "all real
pairs" — its whole job is to keep the holdout untouchable. That is the right
default for training, but it means this evaluation needs its own loader. This
one is read-only: it never carves, never samples, never writes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from twotower.data import LabeledPair, load_canonical_pairs

from eval_real_full.freeze import MANIFEST_PATH, collect, is_real, pair_id, pair_key

Subset = Literal["all", "train", "holdout"]


@dataclass(frozen=True)
class RealPairSet:
    pairs: list[LabeledPair]
    subset: Subset
    combined_hash: str
    n_pos: int
    n_neg: int
    n_candidates: int


def _n_unique_candidates(pairs: list[LabeledPair]) -> int:
    return len({p.pair["matchContactId"] for p in pairs})


def load_real_pairs(
    data_dir: Path,
    split_path: Path,
    *,
    subset: Subset = "all",
    verify: bool = True,
) -> RealPairSet:
    """Return the real pairs for ``subset``, ordered deterministically by pair id.

    ``verify=True`` fails loudly if ``data/`` has drifted from the frozen
    manifest, so a run can never silently score a different population than the
    one this experiment documented.
    """
    current = collect(data_dir, split_path)
    if verify:
        if not MANIFEST_PATH.exists():
            raise FileNotFoundError(
                f"no frozen manifest at {MANIFEST_PATH}; run `python -m eval_real_full.freeze` first"
            )
        frozen = json.loads(MANIFEST_PATH.read_text())
        if frozen["combined_hash"] != current["combined_hash"]:
            raise ValueError(
                f"real pair data drifted since import: frozen={frozen['combined_hash']} "
                f"current={current['combined_hash']}. Run "
                f"`python -m eval_real_full.freeze --verify` to see what changed."
            )

    # Keyed by the opaque pair_key, since the manifest never stores real ids.
    subset_of = {e["pair_key"]: e["subset"] for e in current["pairs"]}

    positives, negatives = load_canonical_pairs(data_dir)
    pairs: list[LabeledPair] = []
    for label, rows in (("pos", positives), ("neg", negatives)):
        for pair in rows:
            if not is_real(pair):
                continue
            pid = pair_id(pair, label)
            where = subset_of[pair_key(pid)]
            if subset != "all" and where != subset:
                continue
            pairs.append(
                LabeledPair(
                    pair_id=pid,
                    label=label,  # type: ignore[arg-type]
                    pair=pair,
                    source="real_holdout" if where == "holdout" else "real_train",
                )
            )

    pairs.sort(key=lambda p: p.pair_id)
    return RealPairSet(
        pairs=pairs,
        subset=subset,
        combined_hash=current["combined_hash"],
        n_pos=sum(1 for p in pairs if p.label == "pos"),
        n_neg=sum(1 for p in pairs if p.label == "neg"),
        n_candidates=_n_unique_candidates(pairs),
    )
