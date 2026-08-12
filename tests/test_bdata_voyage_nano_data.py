"""Tests for bdata_voyage_nano data loading, split provenance, and encoder drift."""

from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

import numpy as np
import pytest

from bdata_voyage_nano.data import (
    assert_source_locked,
    build_seeker_disjoint_split,
    expand_resolved_pairs,
    partition_pairs,
    verify_split,
)
from bdata_voyage_nano.encode import cosine_scores as bdata_cosine

ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = ROOT / "bdata_voyage_nano" / "split.json"
TFIDF_SPLIT_PATH = ROOT / "bdata_tfidf" / "split.json"


def _tiny_rows() -> list[dict]:
    return [
        {
            "contactId": "seeker_a",
            "query": "find ai investors",
            "contactFile": {"positioning": "founder building X", "lookingFor": "investors"},
            "matches": [
                {
                    "status": "ACCEPT",
                    "matchType": "SMS",
                    "contactFile": {"positioning": "angel investor", "background": "ex-faang"},
                },
                {
                    "status": "PENDING",
                    "matchType": "BACKGROUND_MATCH",
                    "contactFile": {"positioning": "should be dropped"},
                },
                {
                    "status": "REJECT",
                    "matchType": "SMS",
                    "contactFile": {"positioning": "wrong stage founder"},
                },
            ],
        },
        {
            "contactId": "seeker_b",
            "query": "hire designers",
            "contactFile": {"positioning": "startup ceo"},
            "matches": [
                {
                    "status": "ACCEPT",
                    "matchType": "ON_CALL",
                    "contactFile": {"positioning": "product designer"},
                },
            ],
        },
        {
            "contactId": "seeker_c",
            "query": "partnerships",
            "contactFile": {"positioning": "bizdev"},
            "matches": [
                {
                    "status": "REJECT",
                    "matchType": "BACKGROUND_MATCH",
                    "contactFile": {"positioning": "competitor"},
                },
            ],
        },
    ]


def test_expand_drops_pending_keeps_accept_reject():
    pairs = expand_resolved_pairs(_tiny_rows())
    assert len(pairs) == 4
    assert {p.label for p in pairs} == {"ACCEPT", "REJECT"}


def test_seeker_disjoint_split_no_overlap():
    pairs = expand_resolved_pairs(_tiny_rows())
    split = build_seeker_disjoint_split(pairs, holdout_frac=0.34, seed=7)
    assert not (set(split["train_seeker_ids"]) & set(split["holdout_seeker_ids"]))
    verify_split(split, pairs)
    train_pairs, hold_pairs = partition_pairs(pairs, split)
    assert {p.seeker_id for p in train_pairs}.isdisjoint({p.seeker_id for p in hold_pairs})


def test_assert_source_locked(tmp_path: Path):
    path = tmp_path / "B-data.json"
    path.write_text("[]", encoding="utf-8")
    path.chmod(stat.S_IWUSR | stat.S_IRUSR)
    with pytest.raises(PermissionError, match="writable"):
        assert_source_locked(path)
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    assert_source_locked(path)


def test_cosine_scores_matches_baselines_voyage_nano():
    from baselines.voyage_nano.encode import cosine_scores as base_cosine

    rng = np.random.default_rng(0)
    a = rng.normal(size=(8, 16)).astype(np.float32)
    b = rng.normal(size=(8, 16)).astype(np.float32)
    a /= np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
    b /= np.linalg.norm(b, axis=1, keepdims=True) + 1e-12
    np.testing.assert_allclose(bdata_cosine(a, b), base_cosine(a, b), rtol=0, atol=0)


def test_split_matches_bdata_tfidf_byte_identical():
    """Matched-population rule: same freeze as the TF-IDF B-data experiment."""
    assert SPLIT_PATH.is_file()
    assert TFIDF_SPLIT_PATH.is_file()
    ours = SPLIT_PATH.read_bytes()
    theirs = TFIDF_SPLIT_PATH.read_bytes()
    assert hashlib.sha256(ours).hexdigest() == hashlib.sha256(theirs).hexdigest()
    split = json.loads(ours)
    assert split["split_hash"].startswith("0f050493daf9")
    assert split["n_holdout_pairs"] == 5585
