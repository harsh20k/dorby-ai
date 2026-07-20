"""Unit tests for leakage-safe two-tower data loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from twotower.data import (
    assert_no_holdout_leak,
    build_split_bundle,
    load_canonical_pairs,
    pairs_to_hf_dict,
    to_triplet_rows,
)
from synth_pipeline.split import load_split

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SPLIT_PATH = DATA_DIR / "synthetic" / "seed_split.json"


@pytest.fixture(scope="module")
def bundle():
    return build_split_bundle(DATA_DIR, SPLIT_PATH)


def test_canonical_counts():
    pos, neg = load_canonical_pairs(DATA_DIR)
    assert len(pos) == 320
    assert len(neg) == 340


def test_bundle_totals(bundle):
    assert bundle.counts["holdout"] == 69
    assert bundle.counts["train"] + bundle.counts["train_dev"] == 591
    assert (
        bundle.counts["train"]
        + bundle.counts["train_dev"]
        + bundle.counts["holdout"]
        == 660
    )
    assert bundle.excluded_eval_leak == 0
    assert bundle.split_hash == "20bbe8f293127372"


def test_no_holdout_leak(bundle):
    split = load_split(SPLIT_PATH)
    assert_no_holdout_leak(bundle, eval_user_ids=set(split["eval_user_ids"]))


def test_holdout_matches_frozen_eval_ids(bundle):
    split = load_split(SPLIT_PATH)
    assert {p.pair_id for p in bundle.holdout} == set(split["eval_pair_ids"])


def test_train_dev_user_disjoint(bundle):
    train_users = {p.pair["userContactId"] for p in bundle.train}
    dev_users = {p.pair["userContactId"] for p in bundle.train_dev}
    assert not (train_users & dev_users)


def test_carve_deterministic():
    pos, neg = load_canonical_pairs(DATA_DIR)
    b1 = build_split_bundle(DATA_DIR, SPLIT_PATH, seed=42)
    b2 = build_split_bundle(DATA_DIR, SPLIT_PATH, seed=42)
    assert [p.pair_id for p in b1.train_dev] == [p.pair_id for p in b2.train_dev]
    assert [p.pair_id for p in b1.train] == [p.pair_id for p in b2.train]
    del pos, neg


def test_hf_dict_labels(bundle):
    d = pairs_to_hf_dict(bundle.train[:10])
    assert set(d) == {"anchor", "other", "label"}
    assert all(y in (0.0, 1.0) for y in d["label"])
    assert all(isinstance(a, str) and a for a in d["anchor"])


def test_triplet_path_does_not_crash(bundle):
    pos = [p for p in bundle.train if p.label == "pos"]
    neg = [p for p in bundle.train if p.label == "neg"]
    rows = to_triplet_rows(pos, neg)
    assert isinstance(rows, list)


def test_real_only_excludes_synth():
    b = build_split_bundle(DATA_DIR, SPLIT_PATH, include_synth=False)
    assert b.counts["train_synth"] == 0
    assert b.counts["train_dev_synth"] == 0
    assert b.counts["holdout"] == 69  # holdout is always 100% real, unaffected


def test_split_hash_tamper_rejected(tmp_path: Path):
    raw = json.loads(SPLIT_PATH.read_text())
    raw["split_hash"] = "deadbeefdeadbeef"
    bad = tmp_path / "seed_split.json"
    bad.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="split hash mismatch"):
        load_split(bad)
