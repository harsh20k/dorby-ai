"""Leakage-safe train / train-dev / holdout loaders for two-tower fine-tuning."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from baselines.bert_frozen.text import candidate_to_text, seeker_to_text
from synth_pipeline.split import index_seed_pairs, load_split

LabelName = Literal["pos", "neg"]


@dataclass(frozen=True)
class LabeledPair:
    pair_id: str
    label: LabelName
    pair: dict[str, Any]
    source: Literal["real_train", "real_holdout", "synth"]

    @property
    def y(self) -> int:
        return 1 if self.label == "pos" else 0

    @property
    def seeker_text(self) -> str:
        return seeker_to_text(self.pair["userContactFile"], self.pair["searchQuery"])

    @property
    def candidate_text(self) -> str:
        return candidate_to_text(self.pair["matchContactFile"])

    @property
    def user_ids(self) -> set[str]:
        return {self.pair["userContactId"], self.pair["matchContactId"]}


@dataclass(frozen=True)
class SplitBundle:
    train: list[LabeledPair]
    train_dev: list[LabeledPair]
    holdout: list[LabeledPair]
    split_hash: str
    data_hash: str
    excluded_eval_leak: int
    counts: dict[str, int]


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"expected list in {path}")
    return data


def _pair_id(pair: dict[str, Any], label: LabelName) -> str:
    return f"{label}:{pair['userContactId']}:{pair['matchContactId']}"


def _is_synth_pair(pair: dict[str, Any]) -> bool:
    return str(pair["userContactId"]).startswith("cmsynth") or str(
        pair["matchContactId"]
    ).startswith("cmsynth")


def _hash_pairs(pairs: list[LabeledPair]) -> str:
    material = sorted(p.pair_id for p in pairs)
    blob = json.dumps(material, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def load_canonical_pairs(
    data_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pos_path = data_dir / "dataset_positive.json"
    neg_path = data_dir / "dataset_negative.json"
    if not pos_path.exists() or not neg_path.exists():
        raise FileNotFoundError(
            f"Expected {pos_path.name} and {neg_path.name} under {data_dir}"
        )
    return _load_json_list(pos_path), _load_json_list(neg_path)


def build_split_bundle(
    data_dir: Path,
    split_path: Path,
    *,
    train_dev_user_fraction: float = 0.1,
    train_dev_min_pairs: int = 20,
    seed: int = 42,
) -> SplitBundle:
    """Build leakage-safe train / train-dev / holdout from frozen seed_split + synth.

    Holdout = frozen eval_pair_ids only (never used for model selection).
    Train pool = frozen train_pair_ids + promoted synth pairs that touch no eval user.
    Train-dev = user-disjoint carve from the train pool.
    """
    positives, negatives = load_canonical_pairs(data_dir)
    split = load_split(split_path)
    eval_users = set(split["eval_user_ids"])
    train_pair_ids = set(split["train_pair_ids"])
    eval_pair_ids = set(split["eval_pair_ids"])

    index = index_seed_pairs(positives, negatives)
    holdout: list[LabeledPair] = []
    for pid in sorted(eval_pair_ids):
        if pid not in index:
            raise KeyError(f"holdout pair missing from canonical data: {pid}")
        rec = index[pid]
        holdout.append(
            LabeledPair(
                pair_id=pid,
                label=rec["label"],
                pair=rec["pair"],
                source="real_holdout",
            )
        )

    train_pool: list[LabeledPair] = []
    excluded_eval_leak = 0
    seen: set[str] = set()

    for label, rows in (("pos", positives), ("neg", negatives)):
        for pair in rows:
            pid = _pair_id(pair, label)
            if pid in seen:
                continue
            seen.add(pid)

            if pid in eval_pair_ids:
                continue

            contacts = {pair["userContactId"], pair["matchContactId"]}
            if contacts & eval_users:
                # Promoted synth or stray real pair that touches a holdout user.
                excluded_eval_leak += 1
                continue

            if pid in train_pair_ids:
                source: Literal["real_train", "synth"] = "real_train"
            elif _is_synth_pair(pair):
                source = "synth"
            else:
                # Real pair not in frozen train/eval — treat as train only if
                # it does not touch eval users (already checked).
                source = "real_train"

            train_pool.append(
                LabeledPair(pair_id=pid, label=label, pair=pair, source=source)
            )

    train, train_dev = carve_train_dev(
        train_pool,
        user_fraction=train_dev_user_fraction,
        min_pairs=train_dev_min_pairs,
        seed=seed,
    )

    all_for_hash = train + train_dev + holdout
    counts = {
        "canonical_pos": len(positives),
        "canonical_neg": len(negatives),
        "train": len(train),
        "train_dev": len(train_dev),
        "holdout": len(holdout),
        "train_real": sum(1 for p in train if p.source == "real_train"),
        "train_synth": sum(1 for p in train if p.source == "synth"),
        "train_dev_real": sum(1 for p in train_dev if p.source == "real_train"),
        "train_dev_synth": sum(1 for p in train_dev if p.source == "synth"),
        "train_pos": sum(1 for p in train if p.label == "pos"),
        "train_neg": sum(1 for p in train if p.label == "neg"),
        "excluded_eval_leak": excluded_eval_leak,
    }
    return SplitBundle(
        train=train,
        train_dev=train_dev,
        holdout=holdout,
        split_hash=split["split_hash"],
        data_hash=_hash_pairs(all_for_hash),
        excluded_eval_leak=excluded_eval_leak,
        counts=counts,
    )


def carve_train_dev(
    train_pool: list[LabeledPair],
    *,
    user_fraction: float = 0.1,
    min_pairs: int = 20,
    seed: int = 42,
) -> tuple[list[LabeledPair], list[LabeledPair]]:
    """Carve a user-disjoint train-dev slice from the training pool.

    Uses seeker (userContactId) as the unit so the same seeker never appears in
    both train and train-dev.
    """
    if not train_pool:
        return [], []

    by_user: dict[str, list[LabeledPair]] = {}
    for item in train_pool:
        by_user.setdefault(item.pair["userContactId"], []).append(item)

    users = sorted(by_user)
    rng = random.Random(seed)
    rng.shuffle(users)

    n_hold_users = max(1, int(round(len(users) * user_fraction)))
    # Grow user set until we have at least min_pairs (or run out of users).
    dev_users: set[str] = set()
    for uid in users:
        if len(dev_users) >= n_hold_users:
            # Keep adding only if still under min_pairs.
            n_pairs = sum(len(by_user[u]) for u in dev_users)
            if n_pairs >= min_pairs:
                break
        if len(dev_users) >= max(1, len(users) // 2):
            break
        dev_users.add(uid)

    train: list[LabeledPair] = []
    train_dev: list[LabeledPair] = []
    for item in train_pool:
        if item.pair["userContactId"] in dev_users:
            train_dev.append(item)
        else:
            train.append(item)

    # Safety: never empty the train set.
    if not train and train_dev:
        # Move half the users back.
        move = sorted(dev_users)[: max(1, len(dev_users) // 2)]
        move_set = set(move)
        train = [p for p in train_dev if p.pair["userContactId"] in move_set]
        train_dev = [p for p in train_dev if p.pair["userContactId"] not in move_set]

    train_users = {p.pair["userContactId"] for p in train}
    dev_seekers = {p.pair["userContactId"] for p in train_dev}
    if train_users & dev_seekers:
        raise RuntimeError("train/train-dev seeker overlap after carve")

    return sorted(train, key=lambda p: p.pair_id), sorted(
        train_dev, key=lambda p: p.pair_id
    )


def assert_no_holdout_leak(
    bundle: SplitBundle,
    *,
    eval_user_ids: set[str] | None = None,
    split_path: Path | None = None,
) -> None:
    """Raise if train/train-dev contain holdout pair-ids or frozen eval users.

    Note: holdout *pairs* may include train-side users (a train user linked to an
    eval user). Leakage is defined against frozen ``eval_user_ids``, not against
    every contact that appears in a holdout pair.
    """
    if eval_user_ids is None:
        if split_path is None:
            raise ValueError("eval_user_ids or split_path required")
        eval_user_ids = set(load_split(split_path)["eval_user_ids"])

    holdout_ids = {p.pair_id for p in bundle.holdout}
    for name, rows in (("train", bundle.train), ("train_dev", bundle.train_dev)):
        for item in rows:
            if item.pair_id in holdout_ids:
                raise RuntimeError(f"{name} contains holdout pair_id {item.pair_id}")
            leak = item.user_ids & eval_user_ids
            if leak:
                raise RuntimeError(f"{name} touches frozen eval user(s) {sorted(leak)}")


def pairs_to_hf_dict(pairs: list[LabeledPair]) -> dict[str, list[Any]]:
    """Column layout for OnlineContrastiveLoss: (anchor, other, label)."""
    return {
        "anchor": [p.seeker_text for p in pairs],
        "other": [p.candidate_text for p in pairs],
        "label": [float(p.y) for p in pairs],
    }


def to_triplet_rows(
    positives: list[LabeledPair],
    negatives: list[LabeledPair],
) -> list[dict[str, str]]:
    """Optional future path: same-seeker (anchor, positive, negative) triples.

    Returns rows only when a positive and negative share the exact
    (userContactId, searchQuery). Current Boardy data usually yields zero rows.
    """
    neg_by_key: dict[tuple[str, str], list[LabeledPair]] = {}
    for n in negatives:
        key = (n.pair["userContactId"], n.pair["searchQuery"])
        neg_by_key.setdefault(key, []).append(n)

    rows: list[dict[str, str]] = []
    for p in positives:
        key = (p.pair["userContactId"], p.pair["searchQuery"])
        for n in neg_by_key.get(key, []):
            rows.append(
                {
                    "anchor": p.seeker_text,
                    "positive": p.candidate_text,
                    "negative": n.candidate_text,
                }
            )
    return rows
