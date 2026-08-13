"""Tests for bdata_voyage_nano_posbg: lock, packing, ids, encoder drift, no split."""

from __future__ import annotations

import hashlib
import importlib.util
import stat
from pathlib import Path

import numpy as np
import pytest

from bdata_voyage_nano_posbg.config import ExperimentConfig
from bdata_voyage_nano_posbg.data import (
    EMPTY_CONTACT_ID,
    assert_source_locked,
    build_id_map,
    expand_resolved_pairs,
    identity_key,
    mint_id,
    retrieval_corpus,
)
from bdata_voyage_nano_posbg.encode import cosine_scores as posbg_cosine
from bdata_voyage_nano_posbg.eval import batched_retrieval_ranks
from bdata_voyage_nano_posbg.text import bg_text, seeker_look_text

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "build_unique_contacts_B_data.py"


def _tiny_rows() -> list[dict]:
    return [
        {
            "contactId": "cm" + "a" * 25,
            "query": "find ai investors",
            "contactFile": {
                "positioning": "founder building X",
                "background": "ex-pm at a lab",
                "lookingFor": "investors",
                "notes": "should never appear in seeker text",
            },
            "matches": [
                {
                    "status": "ACCEPT",
                    "matchType": "SMS",
                    "contactFile": {
                        "positioning": "angel investor",
                        "background": "ex-faang",
                        "lookingFor": "deal flow",
                        "notes": "should never appear in cand text",
                    },
                },
                {
                    "status": "PENDING",
                    "matchType": "BACKGROUND_MATCH",
                    "contactFile": {"positioning": "should be dropped"},
                },
                {
                    "status": "REJECT",
                    "matchType": "SMS",
                    "contactFile": {
                        "positioning": "wrong stage founder",
                        "background": "pre-idea",
                    },
                },
            ],
        },
        {
            "contactId": "cm" + "b" * 25,
            "query": "hire designers",
            "contactFile": {"positioning": "startup ceo", "lookingFor": "designers"},
            "matches": [
                {
                    "status": "ACCEPT",
                    "matchType": "ON_CALL",
                    "contactFile": {"positioning": "product designer"},
                },
            ],
        },
    ]


def test_expand_drops_pending_keeps_accept_reject():
    pairs = expand_resolved_pairs(_tiny_rows())
    assert len(pairs) == 3
    assert {p.label for p in pairs} == {"ACCEPT", "REJECT"}


def test_assert_source_locked(tmp_path: Path):
    path = tmp_path / "B-data.json"
    path.write_text("[]", encoding="utf-8")
    path.chmod(stat.S_IWUSR | stat.S_IRUSR)
    with pytest.raises(PermissionError, match="writable"):
        assert_source_locked(path)
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    assert_source_locked(path)


def test_no_split_on_config():
    assert "split_path" not in ExperimentConfig.__dataclass_fields__


def test_text_packing_seeker_look_plus_query_not_full_profile():
    rows = _tiny_rows()
    pairs = expand_resolved_pairs(rows)
    accept = next(p for p in pairs if p.label == "ACCEPT" and "investors" in p.query)
    assert "lookingFor: investors" in accept.seeker_text
    assert "Search query: find ai investors" in accept.seeker_text
    assert "founder building X" not in accept.seeker_text
    assert "ex-pm at a lab" not in accept.seeker_text
    assert "should never appear in seeker text" not in accept.seeker_text
    packed = seeker_look_text(rows[0]["contactFile"], rows[0]["query"])
    assert packed == accept.seeker_text


def test_text_packing_candidate_pos_bg_only():
    rows = _tiny_rows()
    pairs = expand_resolved_pairs(rows)
    accept = next(p for p in pairs if p.label == "ACCEPT" and "investors" in p.query)
    assert "positioning: angel investor" in accept.cand_text
    assert "background: ex-faang" in accept.cand_text
    assert "deal flow" not in accept.cand_text
    assert "should never appear in cand text" not in accept.cand_text
    packed = bg_text(rows[0]["matches"][0]["contactFile"])
    assert packed == accept.cand_text


def test_id_minting_seekers_keep_real_candidate_only_minted_stable():
    real_id = "cm" + "z" * 25
    key_seeker = "a" * 64
    key_cand = "b" * 64
    contacts = [
        {
            "identityKey": key_seeker,
            "role": "both",
            "contactIds": [real_id],
            "contactFile": {"positioning": "seeker person"},
        },
        {
            "identityKey": key_cand,
            "role": "candidate",
            "contactFile": {"positioning": "cand only"},
        },
    ]
    m1 = build_id_map(contacts)
    m2 = build_id_map(contacts)
    assert m1[key_seeker].contact_id == real_id
    assert m1[key_seeker].source == "boardy"
    assert m1[key_cand].source == "minted"
    assert m1[key_cand].contact_id == mint_id(key_cand)
    assert m1[key_cand].contact_id == "cmb" + key_cand[:25]
    assert m1[key_cand].contact_id == m2[key_cand].contact_id
    assert m1[key_seeker].contact_id == m2[key_seeker].contact_id


def test_boardy_id_collision_mints_second_person():
    shared = "cm" + "x" * 25
    key_a = "a" * 64
    key_b = "b" * 64
    contacts = [
        {
            "identityKey": key_a,
            "role": "both",
            "contactIds": [shared],
            "contactFile": {"positioning": "version a"},
        },
        {
            "identityKey": key_b,
            "role": "both",
            "contactIds": [shared],
            "contactFile": {"positioning": "version b"},
        },
    ]
    m = build_id_map(contacts)
    assert m[key_a].contact_id == shared
    assert m[key_a].source == "boardy"
    assert m[key_b].source == "minted_collision"
    assert m[key_b].contact_id == mint_id(key_b)
    assert m[key_a].contact_id != m[key_b].contact_id


def test_id_minting_picks_first_sorted_boardy_id():
    key = "c" * 64
    lo = "cm" + "a" * 25
    hi = "cm" + "z" * 25
    m = build_id_map(
        [{"identityKey": key, "role": "both", "contactIds": [hi, lo], "contactFile": {}}]
    )
    assert m[key].contact_id == lo
    assert m[key].source == "boardy"


def test_expand_uses_lookup_and_mints_without_it():
    rows = _tiny_rows()
    ident = identity_key(rows[0]["matches"][0]["contactFile"])
    assert ident is not None
    _field, digest = ident
    assigned = "cm" + "q" * 25
    pairs_lookup = expand_resolved_pairs(rows, {digest: assigned})
    accept = next(p for p in pairs_lookup if p.label == "ACCEPT" and "investors" in p.query)
    assert accept.match_contact_id == assigned
    assert accept.seeker_id == "cm" + "a" * 25

    pairs_mint = expand_resolved_pairs(rows)
    accept_m = next(p for p in pairs_mint if p.label == "ACCEPT" and "investors" in p.query)
    assert accept_m.match_contact_id == mint_id(digest)


def test_retrieval_corpus_excludes_seeker_only():
    key_both = "1" * 64
    key_cand = "2" * 64
    key_seek = "3" * 64
    contacts = [
        {
            "identityKey": key_both,
            "role": "both",
            "contactIds": ["cm" + "1" * 25],
            "contactFile": {"positioning": "both person", "background": "bg"},
        },
        {
            "identityKey": key_cand,
            "role": "candidate",
            "contactFile": {"positioning": "cand only"},
        },
        {
            "identityKey": key_seek,
            "role": "seeker",
            "contactIds": ["cm" + "3" * 25],
            "contactFile": {"positioning": "seeker only", "lookingFor": "x"},
        },
    ]
    id_map = build_id_map(contacts)
    ids, texts = retrieval_corpus(contacts, id_map)
    assert id_map[key_both].contact_id in ids
    assert id_map[key_cand].contact_id in ids
    assert id_map[key_seek].contact_id not in ids
    assert len(ids) == 2
    assert any("both person" in t for t in texts)
    assert all("lookingFor" not in t for t in texts)


def test_empty_profile_gets_sentinel_id():
    rows = [
        {
            "contactId": "cm" + "e" * 25,
            "query": "q",
            "contactFile": {"lookingFor": "x"},
            "matches": [{"status": "ACCEPT", "contactFile": {}}],
        }
    ]
    pairs = expand_resolved_pairs(rows)
    assert pairs[0].match_contact_id == EMPTY_CONTACT_ID
    assert pairs[0].identity_key is None


def test_cosine_scores_matches_baselines_voyage_nano():
    from baselines.voyage_nano.encode import cosine_scores as base_cosine

    rng = np.random.default_rng(0)
    a = rng.normal(size=(8, 16)).astype(np.float32)
    b = rng.normal(size=(8, 16)).astype(np.float32)
    a /= np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
    b /= np.linalg.norm(b, axis=1, keepdims=True) + 1e-12
    np.testing.assert_allclose(posbg_cosine(a, b), base_cosine(a, b), rtol=0, atol=0)


def test_identity_key_matches_unique_contacts_script():
    spec = importlib.util.spec_from_file_location("uc_script", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cf = {"positioning": "hello", "background": "world", "lookingFor": "x"}
    assert identity_key(cf) == mod.identity_key(cf)
    assert identity_key({}) is None
    assert mod.identity_key({}) is None


def test_batched_ranks_match_retrieval_metrics():
    from baselines.metrics import retrieval_metrics, retrieval_metrics_from_ranks

    rng = np.random.default_rng(0)
    cand = rng.normal(size=(20, 8)).astype(np.float32)
    cand /= np.linalg.norm(cand, axis=1, keepdims=True) + 1e-12
    query = cand[[0, 3, 7]]
    ids = [f"c{i}" for i in range(20)]
    targets = ["c0", "c3", "c7"]
    ranks = batched_retrieval_ranks(
        query, targets, ids, cand, query_batch_size=2
    )
    assert ranks == [1, 1, 1]
    m1 = retrieval_metrics(query, targets, ids, cand)
    m2 = retrieval_metrics_from_ranks(ranks)
    assert m1["mrr"] == pytest.approx(m2["mrr"])
    assert m1["recall@1"] == pytest.approx(m2["recall@1"])


def test_previous_experiment_files_untouched():
    """Isolation: we must not have edited the prior nano B-data package."""
    prior = ROOT / "bdata_voyage_nano" / "encode.py"
    assert prior.is_file()
    text = prior.read_text(encoding="utf-8")
    assert "artifacts/bdata_voyage_nano\"" in text or "artifacts/bdata_voyage_nano')" in text
    assert "bdata_voyage_nano_posbg" not in text
