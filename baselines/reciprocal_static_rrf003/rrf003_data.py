"""Load rrf_003's judge-labeled synthetic pairs as ``LabeledPair``s.

Read-only: loads from the git-tracked, already-frozen export
``exports/rrf_datasets/rrf_003/`` (batch manifest + staged pair JSON files).
Never writes back, never promotes into ``data/dataset_*.json`` — these are a
model's opinion (``google/gemini-3.1-flash-lite`` judge), not real
accept/decline outcomes; see ``exports/rrf_datasets/rrf_003/README.md``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from twotower.data import LabeledPair

DEFAULT_RRF003_DIR = Path("exports/rrf_datasets/rrf_003")


@dataclass(frozen=True)
class Rrf003Set:
    pairs: list[LabeledPair]
    n_pos: int
    n_neg: int
    n_candidates: int
    batch_id: str
    split_hash: str


def load_rrf003_pairs(rrf003_dir: Path = DEFAULT_RRF003_DIR) -> Rrf003Set:
    manifest = json.loads((rrf003_dir / "manifest.json").read_text())
    pairs: list[LabeledPair] = []
    for record in manifest["records"]:
        staged_path = rrf003_dir / record["path"]
        staged = json.loads(staged_path.read_text())
        pair_dict: dict[str, Any] = staged["pair"]
        pairs.append(
            LabeledPair(
                pair_id=record["pair_key"],
                label=record["label"],
                pair=pair_dict,
                source="synth",
            )
        )
    n_pos = sum(1 for p in pairs if p.label == "pos")
    n_neg = sum(1 for p in pairs if p.label == "neg")
    n_candidates = len({p.pair["matchContactId"] for p in pairs})
    return Rrf003Set(
        pairs=pairs,
        n_pos=n_pos,
        n_neg=n_neg,
        n_candidates=n_candidates,
        batch_id=manifest["batch_id"],
        split_hash=manifest["split_hash"],
    )
