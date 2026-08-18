"""Tests for bdata_queryonly_back_look_frozen: lock, packing, no adapter, no split."""

from __future__ import annotations

import ast
import importlib.util
import stat
from pathlib import Path

import numpy as np
import pytest

from bdata_queryonly_back_look_frozen.config import ExperimentConfig
from bdata_queryonly_back_look_frozen.data import (
    EMPTY_CONTACT_ID,
    assert_source_locked,
    build_id_map,
    expand_resolved_pairs,
    identity_key,
    mint_id,
    retrieval_corpus,
)
from bdata_queryonly_back_look_frozen.encode import cosine_scores as frozen_cosine
from bdata_queryonly_back_look_frozen.eval import batched_retrieval_ranks
from bdata_queryonly_back_look_frozen.text import background_lookingfor, query_only

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "build_unique_contacts_B_data.py"
PKG = ROOT / "bdata_queryonly_back_look_frozen"
PRIOR_LORA = ROOT / "bdata_queryonly_back_look"


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


def test_no_adapter_on_config():
    assert "adapter_dir" not in ExperimentConfig.__dataclass_fields__
    artifacts = str(ExperimentConfig().artifacts_dir)
    assert artifacts.endswith("bdata_queryonly_back_look_frozen")
    assert "bdata_queryonly_back_look/" not in artifacts + "/"


def test_encoder_does_not_load_adapter():
    encode_src = (PKG / "encode.py").read_text(encoding="utf-8")
    assert "load_adapter" not in encode_src
    tree = ast.parse(encode_src)
    init = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "VoyageNanoEncoder":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    init = item
                    break
    assert init is not None
    arg_names = [a.arg for a in init.args.args]
    assert "adapter_dir" not in arg_names


def test_eval_does_not_require_adapter():
    eval_src = (PKG / "eval.py").read_text(encoding="utf-8")
    assert "adapter_dir" not in eval_src
    assert "load_adapter" not in eval_src
    modal_src = (PKG / "modal_eval.py").read_text(encoding="utf-8")
    assert "adapter" not in modal_src.lower()
    assert "dorby-bdata-queryonly-back-look-eval" not in modal_src
    assert "dorby-twotower-queryonly-back-look-checkpoints" not in modal_src
    assert "dorby-bdata-queryonly-back-look-frozen" in modal_src


def test_text_packing_seeker_is_query_only():
    rows = _tiny_rows()
    pairs = expand_resolved_pairs(rows)
    accept = next(p for p in pairs if p.label == "ACCEPT" and "investors" in p.query)
    assert accept.seeker_text == "Search query: find ai investors"
    assert "founder building X" not in accept.seeker_text
    assert "lookingFor" not in accept.seeker_text
    assert "should never appear in seeker text" not in accept.seeker_text
    assert query_only(rows[0]["contactFile"], rows[0]["query"]) == accept.seeker_text


def test_text_packing_candidate_background_lookingfor():
    rows = _tiny_rows()
    pairs = expand_resolved_pairs(rows)
    accept = next(p for p in pairs if p.label == "ACCEPT" and "investors" in p.query)
    assert "background: ex-faang" in accept.cand_text
    assert "lookingFor: deal flow" in accept.cand_text
    assert "angel investor" not in accept.cand_text
    assert "should never appear in cand text" not in accept.cand_text
    packed = background_lookingfor(rows[0]["matches"][0]["contactFile"])
    assert packed == accept.cand_text


def test_query_only_matches_shared_builder():
    from query_weighted.text import query_only as shared

    profile = {"positioning": "x", "lookingFor": "y"}
    assert query_only(profile, "find investors") == shared(profile, "find investors")
    assert query_only(profile, "  ") == shared(profile, "  ")


def test_background_lookingfor_matches_shared_builder():
    from field_pairs_sweep.text import background_lookingfor as shared

    profile = {
        "positioning": "should omit",
        "background": "ex-faang",
        "lookingFor": "deal flow",
        "notes": "omit",
    }
    assert background_lookingfor(profile) == shared(profile)


def test_empty_query_falls_back_to_profile():
    from baselines.bert_frozen.text import profile_to_text

    profile = {"positioning": "founder", "lookingFor": "investors"}
    assert query_only(profile, "") == profile_to_text(profile)


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
    assert m1[key_cand].contact_id == m2[key_cand].contact_id


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
    assert m[key_a].source == "boardy"
    assert m[key_b].source == "minted_collision"
    assert m[key_a].contact_id != m[key_b].contact_id


def test_retrieval_corpus_uses_background_lookingfor():
    key_both = "1" * 64
    key_cand = "2" * 64
    key_seek = "3" * 64
    contacts = [
        {
            "identityKey": key_both,
            "role": "both",
            "contactIds": ["cm" + "1" * 25],
            "contactFile": {
                "positioning": "both person",
                "background": "bg both",
                "lookingFor": "asks",
            },
        },
        {
            "identityKey": key_cand,
            "role": "candidate",
            "contactFile": {"background": "cand bg", "lookingFor": "cand look"},
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
    joined = "\n".join(texts)
    assert "bg both" in joined
    assert "asks" in joined
    assert "both person" not in joined


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


def test_cosine_scores_matches_baselines_voyage_nano():
    from baselines.voyage_nano.encode import cosine_scores as base_cosine

    rng = np.random.default_rng(0)
    a = rng.normal(size=(8, 16)).astype(np.float32)
    b = rng.normal(size=(8, 16)).astype(np.float32)
    a /= np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
    b /= np.linalg.norm(b, axis=1, keepdims=True) + 1e-12
    np.testing.assert_allclose(frozen_cosine(a, b), base_cosine(a, b), rtol=0, atol=0)


def test_identity_key_matches_unique_contacts_script():
    spec = importlib.util.spec_from_file_location("uc_script", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cf = {"positioning": "hello", "background": "world", "lookingFor": "x"}
    assert identity_key(cf) == mod.identity_key(cf)


def test_batched_ranks_match_retrieval_metrics():
    from baselines.metrics import retrieval_metrics, retrieval_metrics_from_ranks

    rng = np.random.default_rng(0)
    cand = rng.normal(size=(20, 8)).astype(np.float32)
    cand /= np.linalg.norm(cand, axis=1, keepdims=True) + 1e-12
    query = cand[[0, 3, 7]]
    ids = [f"c{i}" for i in range(20)]
    targets = ["c0", "c3", "c7"]
    ranks = batched_retrieval_ranks(query, targets, ids, cand, query_batch_size=2)
    assert ranks == [1, 1, 1]
    m1 = retrieval_metrics(query, targets, ids, cand)
    m2 = retrieval_metrics_from_ranks(ranks)
    assert m1["mrr"] == pytest.approx(m2["mrr"])


def test_hardness_texts_are_full_profile():
    rows = _tiny_rows()
    pairs = expand_resolved_pairs(rows)
    accept = next(p for p in pairs if p.label == "ACCEPT" and "investors" in p.query)
    assert "founder building X" in accept.hardness_seeker_text
    assert "Search query: find ai investors" in accept.hardness_seeker_text
    assert "angel investor" in accept.hardness_cand_text
    assert accept.hardness_seeker_text != accept.seeker_text
    assert accept.hardness_cand_text != accept.cand_text


def test_previous_experiment_files_untouched():
    prior_lora_encode = PRIOR_LORA / "encode.py"
    assert prior_lora_encode.is_file()
    lora_text = prior_lora_encode.read_text(encoding="utf-8")
    assert "bdata_queryonly_back_look_frozen" not in lora_text
    assert "load_adapter" in lora_text

    for rel in (
        "bdata_voyage_nano_posbg/encode.py",
        "twotower_queryonly_back_look/eval.py",
        "baselines/voyage_nano/encode.py",
        "synth_pipeline/pairing_rrf/store.py",
    ):
        path = ROOT / rel
        assert path.is_file()
        assert "bdata_queryonly_back_look_frozen" not in path.read_text(encoding="utf-8")


def test_unique_first_seen_matches_encode_aligned_order():
    from bdata_queryonly_back_look_frozen.store import expand_unique, unique_first_seen

    texts = ["a", "b", "a", "c", "b"]
    unique, rows = unique_first_seen(texts)
    assert unique == ["a", "b", "c"]
    assert rows == [0, 1, 0, 2, 1]
    mat = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    expanded = expand_unique(mat, rows)
    np.testing.assert_array_equal(expanded.ravel(), [1.0, 2.0, 1.0, 3.0, 2.0])


def test_chroma_roundtrip_isolated_tmp(tmp_path: Path):
    chromadb = pytest.importorskip("chromadb")
    from bdata_queryonly_back_look_frozen.store import (
        CANDIDATE_COLLECTION,
        QUERY_COLLECTION,
        persist_vector_store,
    )

    chroma_dir = tmp_path / "chroma"
    vectors_dir = tmp_path / "vectors"
    corpus_ids = ["c0", "c1", "c2"]
    rng = np.random.default_rng(0)
    corpus = rng.normal(size=(3, 8)).astype(np.float32)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    queries = corpus[[0, 2]]
    query_ids = ["q0", "q1"]

    info = persist_vector_store(
        vectors_dir=vectors_dir,
        chroma_dir=chroma_dir,
        corpus_ids=corpus_ids,
        corpus_emb=corpus,
        corpus_texts=["bg look 0", "bg look 1", "bg look 2"],
        query_ids=query_ids,
        query_emb=queries,
        query_meta=[
            {"seeker_id": "s0", "label": "ACCEPT", "match_contact_id": "c0"},
            {"seeker_id": "s1", "label": "ACCEPT", "match_contact_id": "c2"},
        ],
        query_texts=["Search query: q0", "Search query: q1"],
        model_name="voyageai/voyage-4-nano",
    )
    assert info["backend"] == "chroma"
    assert info["n_candidates"] == 3
    assert info["n_queries"] == 2
    assert (vectors_dir / "corpus.npy").is_file()
    assert (chroma_dir).is_dir()

    client = chromadb.PersistentClient(path=str(chroma_dir))
    cand = client.get_collection(CANDIDATE_COLLECTION)
    qcol = client.get_collection(QUERY_COLLECTION)
    assert cand.count() == 3
    assert qcol.count() == 2
    hit = cand.query(query_embeddings=[corpus[0].tolist()], n_results=1)
    assert hit["ids"][0][0] == "c0"


def test_chroma_rebuild_from_unique_npy(tmp_path: Path):
    pytest.importorskip("chromadb")
    from bdata_queryonly_back_look_frozen.store import (
        persist_vector_store_from_unique_cache,
        unique_first_seen,
    )

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    texts = ["same", "other", "same"]
    ids = ["a", "b", "c"]
    unique, _rows = unique_first_seen(texts)
    unique_mat = np.array(
        [[1.0, 0.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    np.save(artifacts / f"emb_corpus_cand_u{len(unique)}.npy", unique_mat)

    q_texts = ["q", "q"]
    q_unique, _ = unique_first_seen(q_texts)
    q_mat = np.array([[0.0, 1.0]], dtype=np.float32)
    np.save(artifacts / f"emb_all_pos_seeker_u{len(q_unique)}.npy", q_mat)

    info = persist_vector_store_from_unique_cache(
        artifacts_dir=artifacts,
        corpus_ids=ids,
        corpus_texts=texts,
        query_ids=["p0", "p1"],
        query_texts=q_texts,
        query_meta=[{"label": "ACCEPT"}, {"label": "ACCEPT"}],
        model_name="test",
    )
    assert info["n_candidates"] == 3
    loaded = np.load(artifacts / "vectors" / "corpus.npy")
    assert loaded.shape == (3, 2)
    np.testing.assert_array_equal(loaded[0], loaded[2])
    assert not np.allclose(loaded[0], loaded[1])
