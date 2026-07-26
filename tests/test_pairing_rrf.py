"""Tests for the RRF pairing pipeline — no network, no GPU, no paid calls.

Vectors are synthesized so the retrieval chain can be checked end to end without
loading an embedding model: what matters here is the plumbing (which vector maps
to which seeker section, how the two channels fuse, which gates drop what), not
the numbers a real encoder would produce.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from synth_pipeline.pairing.stage import ENVELOPE_KEYS
from synth_pipeline.pairing_rrf import embed as embed_mod
from synth_pipeline.pairing_rrf import fuse as fuse_mod
from synth_pipeline.pairing_rrf import label as label_mod
from synth_pipeline.pairing_rrf import recall as recall_mod
from synth_pipeline.pairing_rrf import store as store_mod
from synth_pipeline.pairing_rrf.sections import (
    WHOLE,
    looking_for_sections,
    query_targets,
    seeker_vectors,
)

TWO_SECTION_PROFILE = {
    "positioning": "Founder of a fintech infrastructure startup",
    "background": "Previously staff engineer at a payments company",
    "lookingFor": (
        "### Fundraising\nSeed investors who write first cheques into fintech.\n\n"
        "### Hiring\nSenior backend engineers with payments experience."
    ),
}
ONE_SECTION_PROFILE = {
    "positioning": "Solo researcher",
    "lookingFor": "Collaborators on marine biology field work.",
}


# --------------------------------------------------------------------------
# sectioning
# --------------------------------------------------------------------------

def test_two_sections_give_n_plus_one_vectors():
    vecs = seeker_vectors("cmA", TWO_SECTION_PROFILE)
    assert [v.key for v in vecs] == ["cmA::whole", "cmA::s0", "cmA::s1"]
    assert vecs[0].is_whole and vecs[0].section_index == WHOLE


def test_section_vector_drops_sibling_sections():
    _, s0, s1 = seeker_vectors("cmA", TWO_SECTION_PROFILE)
    assert "Fundraising" in s0.text and "Hiring" not in s0.text
    assert "Hiring" in s1.text and "Fundraising" not in s1.text
    # Non-lookingFor fields survive untouched in every variant.
    assert "payments company" in s0.text and "payments company" in s1.text


def test_single_section_profile_is_not_duplicated():
    """One section would make the per-section text identical to the whole."""
    assert len(seeker_vectors("cmB", ONE_SECTION_PROFILE)) == 1


def test_profile_without_looking_for_still_gets_a_vector():
    vecs = seeker_vectors("cmC", {"positioning": "Angel investor"})
    assert len(vecs) == 1 and vecs[0].is_whole
    assert query_targets("cmC", {"positioning": "Angel investor"}) == []


def test_one_query_target_per_section():
    targets = query_targets("cmA", TWO_SECTION_PROFILE)
    assert [t.key for t in targets] == ["cmA::q0", "cmA::q1"]
    assert "Fundraising" in targets[0].section_text
    assert len(looking_for_sections(TWO_SECTION_PROFILE)) == 2


# --------------------------------------------------------------------------
# BM25
# --------------------------------------------------------------------------

def test_bm25_ranks_lexical_overlap_first():
    idx = recall_mod.BM25Index(
        ["c1", "c2", "c3"],
        [
            "fintech seed investor writing first cheques",
            "senior backend golang engineer",
            "marine biology coral reef research",
        ],
    )
    hits = idx.top_k("looking for a fintech seed investor", 3)
    assert hits[0].candidate_id == "c1"


def test_bm25_returns_nothing_when_no_term_matches():
    idx = recall_mod.BM25Index(["c1"], ["marine biology research"])
    assert idx.top_k("quantum semiconductor lithography", 3) == []


# --------------------------------------------------------------------------
# fusion
# --------------------------------------------------------------------------

def _recall(dense, lexical):
    from synth_pipeline.pairing_rrf.sections import QueryTarget

    return recall_mod.QueryRecall(
        target=QueryTarget("cmA", 0, "seed investors"),
        query_text="seed investors in fintech",
        dense=recall_mod.ChannelResult("dense", dense),
        lexical=recall_mod.ChannelResult("lexical", lexical),
    )


def test_candidate_found_by_both_channels_outranks_either_alone():
    H = store_mod.Hit
    fused = fuse_mod.rrf_fuse(
        _recall([H("D1", 0.9), H("BOTH", 0.8)], [H("L1", 5.0), H("BOTH", 4.0)])
    )
    assert fused[0].candidate_id == "BOTH"
    assert fused[0].found_by_both


def test_dense_outranks_lexical_at_equal_rank():
    """The 2:1 weight has to actually decide ties, or it is decoration."""
    H = store_mod.Hit
    fused = fuse_mod.rrf_fuse(_recall([H("D1", 0.9)], [H("L1", 5.0)]))
    assert [c.candidate_id for c in fused] == ["D1", "L1"]
    assert fused[0].rrf_score == pytest.approx(2 * fused[1].rrf_score)


def test_equal_weights_restore_the_tie():
    H = store_mod.Hit
    fused = fuse_mod.rrf_fuse(
        _recall([H("D1", 0.9)], [H("L1", 5.0)]), dense_weight=1.0, lexical_weight=1.0
    )
    assert fused[0].rrf_score == pytest.approx(fused[1].rrf_score)


# --------------------------------------------------------------------------
# shortlist gates
# --------------------------------------------------------------------------

def _empty_shortlist_args():
    return {
        "seeker_vectors": np.zeros((0, 4), np.float32),
        "candidate_index": {},
        "candidate_matrix": np.zeros((0, 4), np.float32),
    }


def test_top_k_cut_records_every_drop():
    H = store_mod.Hit
    sl = fuse_mod.build_shortlist(
        _recall([H("a", 0.9), H("b", 0.8), H("c", 0.7)], []),
        top_k=2,
        **_empty_shortlist_args(),
    )
    assert [c.candidate_id for c in sl.candidates] == ["a", "b"]
    assert [d["drop_reason"] for d in sl.dropped] == ["below_top_k"]


def test_similarity_floor_drops_weak_candidates():
    H = store_mod.Hit
    idx = {"a": 0, "b": 1}
    mat = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    seeker = np.array([[1.0, 0.0]], dtype=np.float32)
    sl = fuse_mod.build_shortlist(
        _recall([H("a", 1.0), H("b", 0.0)], []),
        seeker_vectors=seeker,
        candidate_index=idx,
        candidate_matrix=mat,
        top_k=5,
        min_dense_similarity=0.5,
    )
    assert [c.candidate_id for c in sl.candidates] == ["a"]
    assert sl.dropped[0]["drop_reason"] == "below_similarity_floor"


def test_lexical_only_candidate_still_gets_a_dense_score():
    """Otherwise a BM25-only hit sails past the floor untested."""
    H = store_mod.Hit
    sl = fuse_mod.build_shortlist(
        _recall([], [H("b", 4.0)]),
        seeker_vectors=np.array([[1.0, 0.0]], dtype=np.float32),
        candidate_index={"b": 0},
        candidate_matrix=np.array([[0.0, 1.0]], dtype=np.float32),
        top_k=5,
    )
    assert sl.candidates[0].dense_score is not None


def test_seeker_budget_caps_across_sections():
    H = store_mod.Hit
    sls = [
        fuse_mod.build_shortlist(_recall([H("a", 0.9), H("b", 0.8)], []), top_k=2,
                                 **_empty_shortlist_args()),
        fuse_mod.build_shortlist(_recall([H("c", 0.7)], []), top_k=2,
                                 **_empty_shortlist_args()),
    ]
    capped = fuse_mod.apply_seeker_budget(sls, max_pairs_per_seeker=1)
    assert sum(len(s.candidates) for s in capped) == 1
    reasons = {d["drop_reason"] for s in capped for d in s.dropped}
    assert "seeker_budget" in reasons


def test_no_budget_leaves_everything():
    H = store_mod.Hit
    sls = [fuse_mod.build_shortlist(_recall([H("a", 0.9)], []), top_k=2,
                                    **_empty_shortlist_args())]
    assert len(fuse_mod.apply_seeker_budget(sls, max_pairs_per_seeker=None)[0].candidates) == 1


# --------------------------------------------------------------------------
# vector store
# --------------------------------------------------------------------------

def test_exact_store_returns_nearest_first():
    st = store_mod.ExactStore(["a", "b"], np.array([[1, 0], [0, 1]], dtype=np.float32))
    hits = st.query(np.array([[1, 0]], dtype=np.float32), 2)[0]
    assert [h.candidate_id for h in hits] == ["a", "b"]
    assert hits[0].similarity == pytest.approx(1.0)


def test_chroma_and_exact_agree_on_ordering(tmp_path):
    chromadb = pytest.importorskip("chromadb")
    rng = np.random.default_rng(0)
    mat = rng.normal(size=(12, 8)).astype(np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    ids = [f"c{i}" for i in range(12)]
    q = mat[3:4]

    exact = store_mod.ExactStore(ids, mat).query(q, 5)[0]
    chroma = store_mod.build_chroma(tmp_path / "chroma", ids, mat).query(q, 5)[0]
    assert [h.candidate_id for h in exact] == [h.candidate_id for h in chroma]


def test_open_store_honours_no_chroma(tmp_path):
    st = store_mod.open_store(
        tmp_path / "c", ["a"], np.array([[1.0, 0.0]], np.float32), use_chroma=False
    )
    assert isinstance(st, store_mod.ExactStore)


# --------------------------------------------------------------------------
# embedding persistence
# --------------------------------------------------------------------------

def test_persist_then_load_round_trips(tmp_path):
    seekers = {"cmA": TWO_SECTION_PROFILE, "cmB": ONE_SECTION_PROFILE}
    candidates = {"cmX": ONE_SECTION_PROFILE}
    plan = embed_mod.build_plan(seekers, candidates)
    # cmA -> 3 vectors, cmB -> 1
    assert plan.n_seeker == 4 and plan.n_candidate == 1

    smat = np.ones((plan.n_seeker, 6), dtype=np.float32)
    cmat = np.ones((plan.n_candidate, 6), dtype=np.float32)
    embed_mod.persist(tmp_path, plan, smat, cmat, model_name="test/model")

    smat2, cmat2, manifest = embed_mod.load_persisted(tmp_path)
    assert smat2.shape == smat.shape and cmat2.shape == cmat.shape
    assert manifest["n_seekers"] == 2 and manifest["dim"] == 6
    rows = recall_mod.seeker_row_index(manifest)
    assert rows["cmA::whole"] == 0 and rows["cmA::s1"] == 2


def test_plan_row_order_is_deterministic():
    seekers = {"cmB": ONE_SECTION_PROFILE, "cmA": TWO_SECTION_PROFILE}
    a = embed_mod.build_plan(seekers, {})
    b = embed_mod.build_plan(dict(reversed(list(seekers.items()))), {})
    assert [v.key for v in a.seeker_vectors] == [v.key for v in b.seeker_vectors]


def test_cache_name_is_bound_to_text_content():
    """A fixed cache key served stale embeddings once already in this repo."""
    n1 = embed_mod._content_cache_name("p", ["alpha"], "m")
    n2 = embed_mod._content_cache_name("p", ["beta"], "m")
    n3 = embed_mod._content_cache_name("p", ["alpha"], "other-model")
    assert n1 != n2 and n1 != n3


# --------------------------------------------------------------------------
# dense recall wiring
# --------------------------------------------------------------------------

def test_dense_recall_uses_whole_and_matching_section_only():
    from synth_pipeline.pairing_rrf.sections import QueryTarget

    # 4 seeker rows: cmA whole/s0/s1 then cmB whole.
    seeker_mat = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=np.float32
    )
    rows = {"cmA::whole": 0, "cmA::s0": 1, "cmA::s1": 2, "cmB::whole": 3}
    cand = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    st = store_mod.ExactStore(["whole_match", "s0_match", "s1_match"], cand)

    res = recall_mod.dense_recall(st, rows, seeker_mat, QueryTarget("cmA", 0, "x"), k=3)
    best = {h.candidate_id: h.similarity for h in res.hits}
    # Section 0's vector is live, section 1's is not consulted for this query.
    assert best["whole_match"] == pytest.approx(1.0)
    assert best["s0_match"] == pytest.approx(1.0)
    assert best["s1_match"] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# labeling
# --------------------------------------------------------------------------

def _contact_file() -> dict:
    """A complete contact file — ``validate_pair_schema`` requires every field."""
    from synth_pipeline.config import PROFILE_KEYS

    return {key: f"{key} text" for key in PROFILE_KEYS}


def _cid(tag: str) -> str:
    """A real-shaped 25-char ``cm…`` contact id, as the schema insists on."""
    from synth_pipeline.pairing.profiles import contact_id_for

    return contact_id_for("test_run", abs(hash(tag)) % 10_000)


def _judged(match: str, seeker="A", candidate="X", query_key="q0"):
    seeker, candidate = _cid(seeker), _cid(candidate)
    query_key = f"{seeker}::{query_key}"
    contact_file = _contact_file()
    return label_mod.JudgedPair(
        seeker_id=seeker,
        candidate_id=candidate,
        query_key=query_key,
        query_text="find me a fintech seed investor",
        section_index=0,
        seeker_profile=contact_file,
        candidate_profile=contact_file,
        verdict={"match": match, "confidence": 88, "reasoning": "because"},
        fusion={"rrf_score": 0.03},
    )


def test_yes_and_no_both_become_labels():
    assert _judged("yes").label == label_mod.LABEL_POSITIVE
    assert _judged("no").label == label_mod.LABEL_NEGATIVE


def test_missing_verdict_is_excluded_not_guessed():
    j = _judged("yes")
    j.verdict = {}
    assert j.label is None


def test_envelope_matches_the_staging_contract():
    env = label_mod.build_envelope(
        _judged("yes"), batch_id="b", split_hash="h", labeler_meta={"model": "m"}
    )
    assert set(env) == ENVELOPE_KEYS
    # A verdict says a pair is bad, not which axis is bad.
    assert env["failure_mode"] is None
    assert env["label"] == "pos"


def test_confidence_is_recorded_but_named_unused():
    """Stated confidence showed no discrimination; nothing may gate on it."""
    env = label_mod.build_envelope(
        _judged("no"), batch_id="b", split_hash="h", labeler_meta={}
    )
    assert "judge_confidence_unused" in env["qc"]
    assert env["qc"]["judge_confidence_unused"] == 88


def test_write_batch_splits_staged_and_excluded(tmp_path):
    bad = _judged("yes", candidate="Y", query_key="q1")
    bad.verdict = {}
    manifest = label_mod.write_batch(
        tmp_path,
        [_judged("yes"), _judged("no", candidate="Z"), bad],
        batch_id="b",
        split_hash="h",
        labeler_meta={"model": "flash-lite"},
    )
    assert manifest["counts"] == {"pos": 1, "neg": 1, "excluded": 1}
    assert manifest["promoted"] is False
    assert len(list((tmp_path / "staged").glob("*.json"))) == 2
    assert len(list((tmp_path / "excluded").glob("*.json"))) == 1

    written = json.loads(next((tmp_path / "staged").glob("*.json")).read_text())
    assert set(written) == ENVELOPE_KEYS


def test_balance_reports_against_real_data():
    stats = label_mod.label_balance([_judged("yes"), _judged("no", candidate="Z")])
    assert stats["positive"] == 1 and stats["negative"] == 1
    assert stats["edges_per_node"] == pytest.approx(2 / 3, abs=1e-3)
    assert stats["real_reference"]["edges_per_node"] == 0.673


# --------------------------------------------------------------------------
# orchestrator wiring
# --------------------------------------------------------------------------

def test_run_end_to_end_with_stubs(tmp_path, monkeypatch):
    """Exercise run() for real, stubbing only the two paid calls.

    Catches the wiring faults unit tests cannot — wrong row indexing between the
    embedding manifest and the seeker matrix, a query key that does not match the
    section it was generated from, a shortlist built against the wrong matrix.
    """
    from synth_pipeline.pairing.profiles import SynthProfile, contact_id_for
    from synth_pipeline.pairing_rrf import run as run_mod

    profiles = []
    for i in range(12):
        sections = (
            "### Fundraising\nSeed investors in fintech and payments.\n\n"
            "### Hiring\nSenior backend engineers who know distributed systems."
            if i % 2 == 0
            else "Advisors on go-to-market for developer tools."
        )
        prof = {k: f"{k} content for profile {i}" for k in
                ("positioning", "background", "introPreferences", "personalPreferences",
                 "meetingAndSchedulingPreferences", "locationAvailability", "notes")}
        prof["lookingFor"] = sections
        profiles.append(
            SynthProfile(
                contact_id=contact_id_for("stub_run", i),
                profile_id=i,
                archetype="stub",
                profile=prof,
                source_run="stub_run",
            )
        )

    monkeypatch.setattr(run_mod, "load_profile_run", lambda *a, **k: profiles)
    monkeypatch.setattr(run_mod, "_style_examples", lambda *a, **k: ["a real query"])
    monkeypatch.setattr(
        "synth_pipeline.pairing.bedrock.make_client", lambda region: object()
    )

    def fake_queries(targets, profs, **kw):
        return run_mod.query_gen.QueryGenResult(
            queries={t.key: f"looking for {t.section_text[:40]}" for t in targets},
            prompt_ref={"identifier": "-/pair-rrf-query:v1"},
        )

    monkeypatch.setattr(run_mod.query_gen, "generate_queries", fake_queries)

    def fake_embed(plan, **kw):
        rng = np.random.default_rng(7)
        def unit(n):
            m = rng.normal(size=(n, 16)).astype(np.float32)
            return m / np.linalg.norm(m, axis=1, keepdims=True)
        return unit(plan.n_seeker), unit(plan.n_candidate)

    monkeypatch.setattr(run_mod.embed_mod, "embed_plan", fake_embed)

    cfg = run_mod.RunConfig(
        profile_run=tmp_path / "profiles",
        batch_id="stub_batch",
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
        top_k=3,
        recall_k=5,
        skip_judge=True,
        use_chroma=False,
    )
    summary = run_mod.run(cfg)

    out = cfg.batch_dir()
    assert (out / "embeddings" / "seeker_vectors.npy").exists()
    assert (out / "embeddings" / "manifest.json").exists()
    assert (out / "shortlists.json").exists()
    assert (out / "run_summary.json").exists()
    assert summary["pairs_shortlisted"] > 0

    shortlists = json.loads((out / "shortlists.json").read_text())
    assert all(len(s["candidates"]) <= 3 for s in shortlists)
    # A seeker must never be shortlisted against itself: the split is disjoint.
    for s in shortlists:
        assert s["seeker_id"] not in {c["candidate_id"] for c in s["candidates"]}
    # Every query key must name the section it was generated from.
    for s in shortlists:
        assert s["query_key"].endswith(f"q{s['section_index']}")


def test_deduplicate_keeps_strongest_query_per_pair():
    """The same candidate reached by two of a seeker's queries must survive once.

    Left duplicated and judged independently, such pairs came back with
    contradictory labels on the first real run — 25 of them.
    """
    H = store_mod.Hit
    from synth_pipeline.pairing_rrf.sections import QueryTarget

    def sl(section, hits):
        qr = recall_mod.QueryRecall(
            target=QueryTarget("cmA", section, "s"),
            query_text=f"query {section}",
            dense=recall_mod.ChannelResult("dense", hits),
            lexical=recall_mod.ChannelResult("lexical", []),
        )
        return fuse_mod.build_shortlist(qr, top_k=5, **_empty_shortlist_args())

    # "shared" is rank 1 for section 0 and rank 2 for section 1 → section 0 wins.
    out = fuse_mod.deduplicate_pairs([sl(0, [H("shared", 0.9)]),
                                      sl(1, [H("x", 0.9), H("shared", 0.8)])])
    surviving = [(s.section_index, c.candidate_id) for s in out for c in s.candidates]
    assert sorted(surviving) == [(0, "shared"), (1, "x")]
    assert any(d["drop_reason"] == "duplicate_seeker_candidate"
               for s in out for d in s.dropped)


def test_deduplicate_leaves_distinct_pairs_alone():
    H = store_mod.Hit
    from synth_pipeline.pairing_rrf.sections import QueryTarget

    qr = recall_mod.QueryRecall(
        target=QueryTarget("cmA", 0, "s"), query_text="q",
        dense=recall_mod.ChannelResult("dense", [H("a", 0.9), H("b", 0.8)]),
        lexical=recall_mod.ChannelResult("lexical", []),
    )
    sl = fuse_mod.build_shortlist(qr, top_k=5, **_empty_shortlist_args())
    assert len(fuse_mod.deduplicate_pairs([sl])[0].candidates) == 2
