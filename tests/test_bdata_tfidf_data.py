"""Tests for bdata_tfidf data loading, split safety, and encoder drift."""

from __future__ import annotations

import json
import stat
from pathlib import Path

import numpy as np
import pytest

from bdata_tfidf.data import (
    assert_source_locked,
    build_seeker_disjoint_split,
    expand_resolved_pairs,
    partition_pairs,
    verify_split,
)
from bdata_tfidf.encode import cosine_scores as bdata_cosine


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
                {
                    "status": "PENDING",
                    "matchType": "SMS",
                    "contactFile": {"positioning": "also dropped"},
                },
            ],
        },
    ]


def test_expand_drops_pending_keeps_accept_reject():
    pairs = expand_resolved_pairs(_tiny_rows())
    assert len(pairs) == 4
    assert {p.label for p in pairs} == {"ACCEPT", "REJECT"}
    assert sum(1 for p in pairs if p.label == "ACCEPT") == 2
    assert sum(1 for p in pairs if p.label == "REJECT") == 2
    assert all(p.seeker_text for p in pairs)
    assert all(p.cand_text for p in pairs)
    assert all(len(p.match_contact_id) == 64 for p in pairs)  # sha256 hex


def test_seeker_disjoint_split_no_overlap():
    pairs = expand_resolved_pairs(_tiny_rows())
    split = build_seeker_disjoint_split(pairs, holdout_frac=0.34, seed=7)
    train = set(split["train_seeker_ids"])
    holdout = set(split["holdout_seeker_ids"])
    assert not (train & holdout)
    assert train | holdout == {p.seeker_id for p in pairs}
    verify_split(split, pairs)
    train_pairs, hold_pairs = partition_pairs(pairs, split)
    assert {p.seeker_id for p in train_pairs}.isdisjoint({p.seeker_id for p in hold_pairs})
    assert len(train_pairs) + len(hold_pairs) == len(pairs)


def test_split_hash_tamper_rejected():
    pairs = expand_resolved_pairs(_tiny_rows())
    split = build_seeker_disjoint_split(pairs, holdout_frac=0.34, seed=7)
    split["split_hash"] = "deadbeef"
    with pytest.raises(AssertionError, match="split_hash"):
        verify_split(split, pairs)


def test_assert_source_locked(tmp_path: Path):
    path = tmp_path / "B-data.json"
    path.write_text("[]", encoding="utf-8")
    path.chmod(stat.S_IWUSR | stat.S_IRUSR)  # writable
    with pytest.raises(PermissionError, match="writable"):
        assert_source_locked(path)
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    assert_source_locked(path)  # does not raise


def test_cosine_scores_matches_baselines_tfidf():
    from baselines.tfidf.encode import cosine_scores as base_cosine

    rng = np.random.default_rng(0)
    a = rng.normal(size=(8, 16)).astype(np.float32)
    b = rng.normal(size=(8, 16)).astype(np.float32)
    # L2-normalize like TfidfVectorizer output
    a /= np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
    b /= np.linalg.norm(b, axis=1, keepdims=True) + 1e-12
    np.testing.assert_allclose(bdata_cosine(a, b), base_cosine(a, b), rtol=0, atol=0)
