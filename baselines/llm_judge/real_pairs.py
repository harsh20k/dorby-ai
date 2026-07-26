"""Load the 200 real seed pairs (never synthetic) for the LLM-judge experiment.

Real-pair membership comes from ``data/synthetic/seed_split.json``'s
``train_pair_ids`` + ``eval_pair_ids`` (131 + 69 = 200), not from a
``cmsynth`` id-prefix test: the split file is the frozen, content-hashed
record of which pairs are real, and reusing it means ``--split holdout``
scores exactly the same 69-pair population as
docs/baseline-results-holdout.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from synth_pipeline.split import load_split

Split = Literal["all", "train", "holdout"]


def pair_id(pair: dict[str, Any], label: str) -> str:
    """Same ``label:userContactId:matchContactId`` key used by baselines/holdout.py."""
    return f"{label}:{pair['userContactId']}:{pair['matchContactId']}"


def load_real_pairs(
    data_dir: Path,
    split: Split = "all",
    split_path: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (positives, negatives) restricted to real pairs in ``split``."""
    pos_path = data_dir / "dataset_positive.json"
    neg_path = data_dir / "dataset_negative.json"
    if not pos_path.exists() or not neg_path.exists():
        raise FileNotFoundError(
            f"Expected {pos_path.name} and {neg_path.name} under {data_dir}"
        )
    positives = json.loads(pos_path.read_text())
    negatives = json.loads(neg_path.read_text())

    frozen = load_split(split_path or data_dir / "synthetic" / "seed_split.json")
    if split == "train":
        wanted = set(frozen["train_pair_ids"])
    elif split == "holdout":
        wanted = set(frozen["eval_pair_ids"])
    else:
        wanted = set(frozen["train_pair_ids"]) | set(frozen["eval_pair_ids"])

    pos = [p for p in positives if pair_id(p, "pos") in wanted]
    neg = [p for p in negatives if pair_id(p, "neg") in wanted]

    found = len(pos) + len(neg)
    if found != len(wanted):
        raise ValueError(
            f"expected {len(wanted)} real pairs for split={split!r}, found {found} "
            f"({len(pos)} pos / {len(neg)} neg) — canonical dataset may be missing "
            "seed pairs, or split_path is stale"
        )
    return pos, neg
