"""seed_sampler: pick a train-side seed and few-shots; mint synth IDs."""

from __future__ import annotations

import json
import random
from typing import Any

from synth_pipeline.config import FAILURE_MODES, PipelineConfig
from synth_pipeline.ids import new_contact_id
from synth_pipeline.split import assert_train_only, index_seed_pairs, load_split
from synth_pipeline.state import Label, PairState


def _load_pairs(cfg: PipelineConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    positives = json.loads(cfg.positive_path.read_text(encoding="utf-8"))
    negatives = json.loads(cfg.negative_path.read_text(encoding="utf-8"))
    return positives, negatives


def sample_initial_state(
    *,
    cfg: PipelineConfig,
    batch_id: str,
    label: Label,
    failure_mode: str | None = None,
    rng: random.Random | None = None,
    seed_pair_id: str | None = None,
) -> PairState:
    rng = rng or random.Random()
    split = load_split(cfg.split_path)
    positives, negatives = _load_pairs(cfg)
    index = index_seed_pairs(positives, negatives)

    # Prefer same-label seeds for conditioning, fall back to any train pair.
    same_label = [pid for pid in split["train_pair_ids"] if index[pid]["label"] == label]
    pool = same_label or list(split["train_pair_ids"])
    if seed_pair_id is not None:
        if seed_pair_id not in pool:
            raise ValueError(f"seed_pair_id {seed_pair_id} not in train pool for sampling")
        chosen_id = seed_pair_id
    else:
        chosen_id = rng.choice(pool)

    seed_rec = index[chosen_id]
    seed_pair = seed_rec["pair"]
    seed_user_id = seed_pair["userContactId"]

    few_shot_pool = [pid for pid in pool if pid != chosen_id]
    rng.shuffle(few_shot_pool)
    few_shot_pairs = [index[pid]["pair"] for pid in few_shot_pool[: cfg.few_shot_k]]

    if label == "neg":
        mode = failure_mode or rng.choice(FAILURE_MODES)
    else:
        mode = None

    assert_train_only(
        seed_user_id=seed_user_id,
        seed_pair_id=chosen_id,
        few_shot_pairs=few_shot_pairs,
        split=split,
    )

    # Mint IDs up front so generate must use them.
    user_id = new_contact_id()
    match_id = new_contact_id()
    while match_id == user_id:
        match_id = new_contact_id()

    return {
        "label": label,
        "failure_mode": mode,
        "seed_user_id": seed_user_id,
        "seed_pair_id": chosen_id,
        "seed_pair": seed_pair,
        "few_shot_pairs": few_shot_pairs,
        "batch_id": batch_id,
        "split_hash": split["split_hash"],
        "pair": None,
        "metadata": {
            "assigned_user_contact_id": user_id,
            "assigned_match_contact_id": match_id,
            "generate_model": cfg.generate_model,
            "judge_model": cfg.judge_model,
            "prompt_version": cfg.prompt_version,
            "temperature": cfg.temperature,
            "dry_run": cfg.dry_run,
        },
        "qc": {},
        "status": "pending",
        "retry_count": 0,
        "drop_reason": None,
    }


def seed_sampler_node(state: PairState, cfg: PipelineConfig) -> dict[str, Any]:
    """No-op passthrough — sampling happens before graph invoke.

    Kept as a named node so LangSmith spans match the pipeline doc.
    """
    assert_train_only(
        seed_user_id=state["seed_user_id"],
        seed_pair_id=state["seed_pair_id"],
        few_shot_pairs=state["few_shot_pairs"],
        split=load_split(cfg.split_path),
    )
    if state["split_hash"] != load_split(cfg.split_path)["split_hash"]:
        raise RuntimeError("split_hash drifted since batch start — aborting")
    return {"status": "pending"}

