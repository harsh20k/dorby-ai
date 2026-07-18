"""User-disjoint seed split with stable hash for the holdout firewall."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from synth_pipeline.config import HOLDOUT_FRACTION, PipelineConfig


def _load_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"expected list in {path}")
    return data


def _pair_id(pair: dict[str, Any], label: str) -> str:
    return f"{label}:{pair['userContactId']}:{pair['matchContactId']}"


def _all_user_ids(pair: dict[str, Any]) -> set[str]:
    return {pair["userContactId"], pair["matchContactId"]}


def build_user_disjoint_split(
    positives: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
    *,
    holdout_fraction: float = HOLDOUT_FRACTION,
    seed: int = 42,
) -> dict[str, Any]:
    """Hold out ~fraction of *users*, then assign pairs wholly to train or eval.

    A pair is eval if *either* contact is an eval user — so synth never few-shots
    on holdout people. Remaining pairs are train.
    """
    user_to_pairs: dict[str, list[str]] = defaultdict(list)
    pair_records: dict[str, dict[str, Any]] = {}

    for label, rows in (("pos", positives), ("neg", negatives)):
        for pair in rows:
            pid = _pair_id(pair, label)
            pair_records[pid] = {"label": label, "pair": pair}
            for uid in _all_user_ids(pair):
                user_to_pairs[uid].append(pid)

    users = sorted(user_to_pairs)
    rng = random.Random(seed)
    rng.shuffle(users)
    n_holdout = max(1, int(round(len(users) * holdout_fraction)))
    eval_users = set(users[:n_holdout])
    train_users = set(users[n_holdout:])

    train_pair_ids: list[str] = []
    eval_pair_ids: list[str] = []
    for pid, rec in pair_records.items():
        contacts = _all_user_ids(rec["pair"])
        if contacts & eval_users:
            eval_pair_ids.append(pid)
        else:
            train_pair_ids.append(pid)

    payload = {
        "version": 1,
        "seed": seed,
        "holdout_fraction": holdout_fraction,
        "n_users": len(users),
        "n_train_users": len(train_users),
        "n_eval_users": len(eval_users),
        "n_train_pairs": len(train_pair_ids),
        "n_eval_pairs": len(eval_pair_ids),
        "train_user_ids": sorted(train_users),
        "eval_user_ids": sorted(eval_users),
        "train_pair_ids": sorted(train_pair_ids),
        "eval_pair_ids": sorted(eval_pair_ids),
    }
    payload["split_hash"] = _hash_split(payload)
    return payload


def _hash_split(payload: dict[str, Any]) -> str:
    material = {
        "seed": payload["seed"],
        "holdout_fraction": payload["holdout_fraction"],
        "train_user_ids": payload["train_user_ids"],
        "eval_user_ids": payload["eval_user_ids"],
        "train_pair_ids": payload["train_pair_ids"],
        "eval_pair_ids": payload["eval_pair_ids"],
    }
    blob = json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def save_split(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_split(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = _hash_split(payload)
    if payload.get("split_hash") != expected:
        raise ValueError(
            f"split hash mismatch in {path}: stored={payload.get('split_hash')} "
            f"computed={expected}"
        )
    return payload


def ensure_split(cfg: PipelineConfig, *, seed: int = 42, force: bool = False) -> dict[str, Any]:
    if cfg.split_path.exists() and not force:
        return load_split(cfg.split_path)
    positives = _load_json(cfg.positive_path)
    negatives = _load_json(cfg.negative_path)
    payload = build_user_disjoint_split(
        positives,
        negatives,
        holdout_fraction=cfg.holdout_fraction,
        seed=seed,
    )
    save_split(payload, cfg.split_path)
    return payload


def index_seed_pairs(
    positives: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for label, rows in (("pos", positives), ("neg", negatives)):
        for pair in rows:
            out[_pair_id(pair, label)] = {"label": label, "pair": pair}
    return out


def assert_train_only(
    *,
    seed_user_id: str,
    seed_pair_id: str,
    few_shot_pairs: list[dict[str, Any]],
    split: dict[str, Any],
) -> None:
    eval_users = set(split["eval_user_ids"])
    train_pairs = set(split["train_pair_ids"])
    if seed_pair_id not in train_pairs:
        raise RuntimeError(f"holdout leak: seed_pair_id {seed_pair_id} not in train split")
    if seed_user_id in eval_users:
        raise RuntimeError(f"holdout leak: seed_user_id {seed_user_id} is eval")
    for fs in few_shot_pairs:
        contacts = _all_user_ids(fs)
        if contacts & eval_users:
            raise RuntimeError(f"holdout leak: few-shot uses eval user(s) {contacts & eval_users}")
