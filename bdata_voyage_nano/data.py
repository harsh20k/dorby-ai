"""Load locked B-data.json into ACCEPT/REJECT pairs + seeker-disjoint split."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from baselines.bert_frozen.text import candidate_to_text, seeker_to_text

from bdata_voyage_nano.config import DEFAULT_CONFIG, ExperimentConfig

RESOLVED_STATUSES = frozenset({"ACCEPT", "REJECT"})


@dataclass(frozen=True)
class BPair:
    pair_id: str
    seeker_id: str
    match_contact_id: str  # synthetic = sha256(contactFile)
    label: str  # ACCEPT | REJECT
    query: str
    seeker_text: str
    cand_text: str
    match_type: str
    row_index: int
    match_index: int


def assert_source_locked(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing B-data source: {path}")
    if path.stat().st_mode & 0o222:
        raise PermissionError(
            f"Refusing to read writable B-data at {path}. "
            "Lock it first (chmod 444) — source is immutable for this experiment."
        )


def source_provenance(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    st = path.stat()
    return {
        "path": str(path),
        "size_bytes": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _profile_hash(profile: dict[str, Any] | None) -> str:
    payload = profile if isinstance(profile, dict) else {}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def expand_resolved_pairs(rows: list[dict[str, Any]]) -> list[BPair]:
    """One pair per (seeker, query, match) with status ACCEPT or REJECT."""
    pairs: list[BPair] = []
    for row_i, row in enumerate(rows):
        seeker_id = str(row.get("contactId") or "")
        if not seeker_id:
            continue
        query = str(row.get("query") or "")
        seeker_file = row.get("contactFile") if isinstance(row.get("contactFile"), dict) else {}
        seeker_text = seeker_to_text(seeker_file, query)
        matches = row.get("matches") if isinstance(row.get("matches"), list) else []
        for match_i, match in enumerate(matches):
            if not isinstance(match, dict):
                continue
            status = str(match.get("status") or "")
            if status not in RESOLVED_STATUSES:
                continue
            cand_file = (
                match.get("contactFile")
                if isinstance(match.get("contactFile"), dict)
                else {}
            )
            mid = _profile_hash(cand_file)
            pair_id = f"{seeker_id}:{row_i}:{match_i}:{mid[:16]}"
            pairs.append(
                BPair(
                    pair_id=pair_id,
                    seeker_id=seeker_id,
                    match_contact_id=mid,
                    label=status,
                    query=query,
                    seeker_text=seeker_text,
                    cand_text=candidate_to_text(cand_file),
                    match_type=str(match.get("matchType") or "UNKNOWN"),
                    row_index=row_i,
                    match_index=match_i,
                )
            )
    return pairs


def load_pairs(cfg: ExperimentConfig | None = None) -> tuple[list[BPair], dict[str, Any]]:
    cfg = cfg or DEFAULT_CONFIG
    assert_source_locked(cfg.source_path)
    prov = source_provenance(cfg.source_path)
    rows = json.loads(cfg.source_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{cfg.source_path} must be a JSON array")
    pairs = expand_resolved_pairs(rows)
    meta = {
        "provenance": prov,
        "n_rows": len(rows),
        "n_resolved_pairs": len(pairs),
        "n_accept": sum(1 for p in pairs if p.label == "ACCEPT"),
        "n_reject": sum(1 for p in pairs if p.label == "REJECT"),
        "n_unique_seekers": len({p.seeker_id for p in pairs}),
    }
    return pairs, meta


def _split_hash(train_ids: list[str], holdout_ids: list[str], seed: int, frac: float) -> str:
    h = hashlib.sha256()
    h.update(str(seed).encode())
    h.update(b"\0")
    h.update(str(frac).encode())
    h.update(b"\0")
    for sid in train_ids:
        h.update(sid.encode())
        h.update(b"\0")
    for sid in holdout_ids:
        h.update(sid.encode())
        h.update(b"\0")
    return h.hexdigest()


def build_seeker_disjoint_split(
    pairs: list[BPair],
    *,
    holdout_frac: float = 0.30,
    seed: int = 42,
) -> dict[str, Any]:
    seekers = sorted({p.seeker_id for p in pairs})
    rng = random.Random(seed)
    shuffled = list(seekers)
    rng.shuffle(shuffled)
    n_holdout = max(1, int(round(len(shuffled) * holdout_frac)))
    # Keep at least one train seeker if possible
    if n_holdout >= len(shuffled) and len(shuffled) > 1:
        n_holdout = len(shuffled) - 1
    holdout_seekers = sorted(shuffled[:n_holdout])
    train_seekers = sorted(shuffled[n_holdout:])
    holdout_set = set(holdout_seekers)
    train_pair_ids = sorted(p.pair_id for p in pairs if p.seeker_id not in holdout_set)
    holdout_pair_ids = sorted(p.pair_id for p in pairs if p.seeker_id in holdout_set)
    payload = {
        "seed": seed,
        "holdout_frac": holdout_frac,
        "n_seekers_total": len(seekers),
        "n_seekers_train": len(train_seekers),
        "n_seekers_holdout": len(holdout_seekers),
        "train_seeker_ids": train_seekers,
        "holdout_seeker_ids": holdout_seekers,
        "train_pair_ids": train_pair_ids,
        "holdout_pair_ids": holdout_pair_ids,
        "n_train_pairs": len(train_pair_ids),
        "n_holdout_pairs": len(holdout_pair_ids),
    }
    payload["split_hash"] = _split_hash(
        train_seekers, holdout_seekers, seed, holdout_frac
    )
    return payload


def write_split(split: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")


def load_split(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing split file {path}. Run: python -m bdata_voyage_nano --init-split"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def verify_split(split: dict[str, Any], pairs: list[BPair]) -> None:
    train = set(split["train_seeker_ids"])
    holdout = set(split["holdout_seeker_ids"])
    if train & holdout:
        raise AssertionError("train/holdout seekers overlap — split corrupted")
    expected = _split_hash(
        list(split["train_seeker_ids"]),
        list(split["holdout_seeker_ids"]),
        int(split["seed"]),
        float(split["holdout_frac"]),
    )
    if split.get("split_hash") != expected:
        raise AssertionError(
            f"split_hash mismatch (got {split.get('split_hash')}, expected {expected})"
        )
    pair_seekers = {p.seeker_id for p in pairs}
    if not holdout.issubset(pair_seekers):
        missing = holdout - pair_seekers
        raise AssertionError(
            f"holdout seekers missing from current pairs (n={len(missing)}); "
            "B-data may have changed — rebuild split"
        )


def partition_pairs(
    pairs: list[BPair], split: dict[str, Any]
) -> tuple[list[BPair], list[BPair]]:
    verify_split(split, pairs)
    holdout = set(split["holdout_seeker_ids"])
    train = [p for p in pairs if p.seeker_id not in holdout]
    eval_pairs = [p for p in pairs if p.seeker_id in holdout]
    return train, eval_pairs


def within_seeker_dual_label_groups(
    pairs: list[BPair],
) -> dict[str, dict[str, list[BPair]]]:
    """Seekers that have at least one ACCEPT and one REJECT in `pairs`."""
    by: dict[str, dict[str, list[BPair]]] = {}
    for p in pairs:
        bucket = by.setdefault(p.seeker_id, {"ACCEPT": [], "REJECT": []})
        bucket[p.label].append(p)
    return {
        sid: g
        for sid, g in by.items()
        if g["ACCEPT"] and g["REJECT"]
    }
