"""Distillation variant of the two-tower LoRA fine-tune.

Same recipe as ``arm_a_real_only`` (real pairs only, ``ContrastiveLoss``) but
the training label for each pair is swapped from the hard 0/1 accept/decline
outcome to the naive LLM judge's continuous confidence-signed score (see
``scripts/build_judge_soft_labels.py``) — a small test of whether the
judge's soft signal (AUC 0.64, the best model in this repo) is a better
training target than the binary label alone.

Monkeypatches ``twotower.train.pairs_to_hf_dict`` for the one call site that
builds the train dataset, rather than duplicating ``run_training`` — the
train-dev evaluator and holdout eval are untouched, so they still score
against the real hard labels exactly like every other arm.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import twotower.train as train_mod
from twotower.config import TrainConfig
from twotower.data import LabeledPair
from twotower.data import pairs_to_hf_dict as hard_pairs_to_hf_dict


def _build_soft_pairs_to_hf_dict(soft_labels: dict[str, float]):
    def _fn(pairs: list[LabeledPair]) -> dict[str, list[Any]]:
        missing = [p.pair_id for p in pairs if p.pair_id not in soft_labels]
        if missing:
            raise KeyError(f"{len(missing)} train pair(s) missing a soft label, e.g. {missing[:3]}")
        out = hard_pairs_to_hf_dict(pairs)
        out["label"] = [float(soft_labels[p.pair_id]) for p in pairs]
        return out

    return _fn


def run_training_distilled(
    cfg: TrainConfig,
    soft_labels_path: Path,
    *,
    dry_run: bool = False,
    resume_from_checkpoint: str | None = None,
) -> dict[str, Any]:
    soft_labels: dict[str, float] = json.loads(Path(soft_labels_path).read_text(encoding="utf-8"))

    original = train_mod.pairs_to_hf_dict
    train_mod.pairs_to_hf_dict = _build_soft_pairs_to_hf_dict(soft_labels)
    try:
        result = train_mod.run_training(
            cfg, dry_run=dry_run, resume_from_checkpoint=resume_from_checkpoint
        )
    finally:
        train_mod.pairs_to_hf_dict = original
    result["distillation"] = {
        "soft_labels_path": str(soft_labels_path),
        "n_soft_labels": len(soft_labels),
    }
    return result
