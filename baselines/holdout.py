"""Filter canonical pos/neg pairs down to the frozen real holdout.

Shared by baselines/{bert_frozen,voyage_nano,voyage_large}/eval.py's
``--holdout-only`` flag so all three (and twotower's eval) score the exact
same 69-pair population — see docs/possible-bugs.md #3.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from synth_pipeline.split import load_split


def _pair_id(pair: dict[str, Any], label: str) -> str:
    return f"{label}:{pair['userContactId']}:{pair['matchContactId']}"


def filter_to_holdout(
    positives: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
    split_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return only the pairs whose pair_id is in the frozen eval_pair_ids set.

    Mirrors twotower/data.py::build_split_bundle's holdout selection so the
    baseline and twotower numbers are computed on an identical population.
    """
    split = load_split(split_path)
    eval_pair_ids = set(split["eval_pair_ids"])

    pos_holdout = [p for p in positives if _pair_id(p, "pos") in eval_pair_ids]
    neg_holdout = [p for p in negatives if _pair_id(p, "neg") in eval_pair_ids]

    found = len(pos_holdout) + len(neg_holdout)
    if found != len(eval_pair_ids):
        raise ValueError(
            f"expected {len(eval_pair_ids)} holdout pairs, found {found} "
            f"({len(pos_holdout)} pos / {len(neg_holdout)} neg) — canonical "
            "dataset may be missing holdout pairs or split_path is stale"
        )
    return pos_holdout, neg_holdout
