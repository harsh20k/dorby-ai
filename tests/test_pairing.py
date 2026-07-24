"""Tests for synth_pipeline.pairing — no LLM, no network."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from synth_pipeline.config import PROFILE_KEYS
from synth_pipeline.ids import is_valid_contact_id
from synth_pipeline.pairing import stage
from synth_pipeline.pairing.label import Thresholds, label_scores
from synth_pipeline.pairing.profiles import (
    PAIRING_ID_PREFIX,
    SynthProfile,
    _extract_profile,
)
from synth_pipeline.pairing.select import _log_spaced_ranks, select_candidates
from synth_pipeline.ids import new_contact_id

ROOT = Path(__file__).resolve().parents[1]


def make_profile(idx: int, *, archetype: str = "Founder", topic: str = "fintech") -> SynthProfile:
    text = f"{topic} operator number {idx} building payments infrastructure"
    return SynthProfile(
        contact_id=new_contact_id(prefix=PAIRING_ID_PREFIX),
        profile_id=idx,
        archetype=archetype,
        profile={k: f"{k} {text}" for k in PROFILE_KEYS},
        source_run="run_test",
    )


class FakeCandidate:
    """Minimal stand-in for select.Candidate (build_envelope is duck-typed)."""

    def __init__(self, seeker, candidate, query="looking for a fintech CTO in NYC"):
        self.seeker = seeker
        self.candidate = candidate
        self.query = query
        self.query_index = 0
        self.cosine = 0.42
        self.rank = 1
        self.band = "near"
        self.same_archetype = seeker.archetype == candidate.archetype


# --- profiles -------------------------------------------------------------


def test_extract_profile_drops_reasoning_and_unknown_fields():
    raw = {k: f"value {k}" for k in PROFILE_KEYS}
    raw["reasoning"] = "this persona is coherent because ..."
    raw["someFutureField"] = "should not survive"

    out = _extract_profile(raw)

    assert set(out) == set(PROFILE_KEYS)
    assert "reasoning" not in out
    assert "someFutureField" not in out


def test_minted_ids_are_valid_and_synth_prefixed():
    p = make_profile(0)
    assert is_valid_contact_id(p.contact_id)
    # Existing filters (build_real_pairs_graph.is_synthetic, twotower._is_synth_pair)
    # test for "cmsynth" — the pairing prefix must stay compatible with them.
    assert p.contact_id.startswith("cmsynth")
    assert p.contact_id.startswith(PAIRING_ID_PREFIX)


# --- selection ------------------------------------------------------------


def test_log_spaced_ranks_are_geometric_then_backfill():
    assert _log_spaced_ranks(19, 5) == [0, 1, 3, 7, 15]
    # Not enough room for the geometric walk: backfill from the top, no dupes.
    ranks = _log_spaced_ranks(4, 4)
    assert sorted(set(ranks)) == ranks
    assert len(ranks) == 4
    assert max(ranks) < 4


def _selection_fixture(n=8):
    topics = ["fintech", "biotech", "climate", "devtools"]
    profiles = [
        make_profile(i, archetype=f"Arch{i % 3}", topic=topics[i % len(topics)])
        for i in range(n)
    ]
    queries = {p.contact_id: [f"seeking {topics[i % len(topics)]} experts"] for i, p in enumerate(profiles)}
    return profiles, queries


def test_select_candidates_has_no_self_pairs_or_duplicates():
    profiles, queries = _selection_fixture()
    cands = select_candidates(profiles, queries, k_per_query=3)

    assert cands
    assert all(c.seeker.contact_id != c.candidate.contact_id for c in cands)
    keys = [(c.seeker.contact_id, c.candidate.contact_id) for c in cands]
    assert len(keys) == len(set(keys))


def test_select_candidates_is_deterministic():
    profiles, queries = _selection_fixture()
    a = select_candidates(profiles, queries, k_per_query=3)
    b = select_candidates(profiles, queries, k_per_query=3)
    assert [(c.seeker.contact_id, c.candidate.contact_id, c.rank) for c in a] == [
        (c.seeker.contact_id, c.candidate.contact_id, c.rank) for c in b
    ]


def test_select_candidates_dedups_across_two_queries_from_one_seeker():
    profiles = [make_profile(i, topic="fintech") for i in range(4)]
    # Both queries are identical, so both would pick the same ranked candidates.
    queries = {profiles[0].contact_id: ["fintech payments", "fintech payments"]}
    cands = select_candidates(profiles, queries, k_per_query=3)
    keys = [(c.seeker.contact_id, c.candidate.contact_id) for c in cands]
    assert len(keys) == len(set(keys))


# --- labeling -------------------------------------------------------------


def test_label_scores_deadband_excludes_the_middle():
    th = Thresholds(center=0.0, lower=-0.5, upper=0.5, margin=0.25, fit_std=2.0,
                    fit_best_f1=0.6)
    labels = label_scores(np.array([-2.0, -0.5, -0.1, 0.0, 0.3, 0.5, 2.0]), th)
    assert labels == ["neg", "neg", None, None, None, "pos", "pos"]


def test_label_scores_is_monotone_in_score():
    th = Thresholds(center=0.0, lower=-1.0, upper=1.0, margin=0.25, fit_std=4.0,
                    fit_best_f1=0.6)
    order = {"neg": 0, None: 1, "pos": 2}
    scores = np.linspace(-5, 5, 50)
    ranks = [order[v] for v in label_scores(scores, th)]
    assert ranks == sorted(ranks)


# --- staging --------------------------------------------------------------


def test_build_pair_passes_the_canonical_schema():
    from synth_pipeline.schema import validate_pair_schema

    cand = FakeCandidate(make_profile(0), make_profile(1))
    pair = stage.build_pair(cand)

    assert validate_pair_schema(pair) == []
    assert "label" not in pair, "membership is the label, as in the real dataset"
    assert "reasoning" not in pair["userContactFile"]
    assert "reasoning" not in pair["matchContactFile"]


def test_build_envelope_key_set_matches_writer():
    """Drift guard: promote.py and the review browser read writer.py's envelope."""
    src = (ROOT / "synth_pipeline" / "nodes" / "writer.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    writer_keys: set[str] | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "envelope" in targets:
                writer_keys = {
                    k.value for k in node.value.keys if isinstance(k, ast.Constant)
                }
                break

    assert writer_keys, "could not locate the envelope dict in writer.py"
    assert stage.ENVELOPE_KEYS == writer_keys

    cand = FakeCandidate(make_profile(0), make_profile(1))
    env = stage.build_envelope(
        cand, batch_id="b", label="pos", score=1.5,
        split_hash="deadbeef", scorer_meta={},
    )
    assert set(env) == writer_keys


def test_build_envelope_leaves_failure_mode_unset():
    """A scorer yields a number, not a diagnosis — don't invent a failure_mode."""
    cand = FakeCandidate(make_profile(0), make_profile(1))
    env = stage.build_envelope(
        cand, batch_id="b", label="neg", score=-1.5,
        split_hash="deadbeef", scorer_meta={},
    )
    assert env["failure_mode"] is None
    assert env["qc"]["fusion_score"] == -1.5
    assert env["metadata"]["labeler"] == "hybrid_tfidf_voyage"


def test_build_envelope_rejects_an_invalid_pair():
    bad = make_profile(0)
    cand = FakeCandidate(bad, bad)  # self-pair -> same id both sides
    cand.query = ""                  # and an empty query
    with pytest.raises(ValueError):
        stage.build_envelope(
            cand, batch_id="b", label="pos", score=1.0,
            split_hash="x", scorer_meta={},
        )
