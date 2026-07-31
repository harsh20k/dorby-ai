"""Tests for the sectioned MoE experiment.

Two jobs: pin the new primitives (section splitting, pooling, grouping), and pin
the *drift guards* the isolation rule asks for — this package imports
``moe_rrf.features`` and ``moe_reranker.diagnostics`` read-only, and those
imports must keep meaning what they meant when the numbers were published.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from moe_sectioned.config import SectionedConfig
from moe_sectioned.data import regroup, rows_for_pairs, seeker_disjoint_folds
from moe_sectioned.encode import TfidfBackend, assert_same_space, text_hash
from moe_sectioned.model import POOLING_MODES, SectionedMoE, balance_loss, sharpen_loss
from moe_sectioned.sections import Section, sections_for_pair, split_looking_for

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

needs_data = pytest.mark.skipif(
    not (DATA_DIR / "dataset_positive.json").exists(),
    reason="data/ is gitignored",
)


# ------------------------------------------------------------------ sections
def test_splits_on_headings_not_blank_lines():
    """The whole point: 4 asks, not 13 paragraph blocks."""
    lf = (
        "### Customers\npara one\n\npara two\n\npara three\n\n"
        "### Hiring\nlooking for senior backend engineers in the US market\n\nmore\n\n"
        "### Partnerships\nSaaS companies for GTM collaboration conversations\n"
    )
    secs = split_looking_for(lf, min_chars=5)
    assert [s.heading for s in secs] == ["Customers", "Hiring", "Partnerships"]


def test_heading_and_body_both_embedded():
    s = Section(index=0, heading="Fundraising", body="seed investors in fintech")
    assert s.text.startswith("Fundraising")
    assert "seed investors" in s.text


def test_no_headings_falls_back_to_whole_field():
    secs = split_looking_for("just one long paragraph about what I want here", min_chars=5)
    assert len(secs) == 1 and secs[0].heading == ""


def test_stubs_dropped_but_never_all_of_them():
    lf = "### A\nx\n### B\ny\n"
    secs = split_looking_for(lf, min_chars=500)
    # Every section is a stub, yet the pair must still produce a row.
    assert len(secs) == 1


def test_cap_limits_the_tail():
    lf = "".join(f"### S{i}\n{'word ' * 20}\n" for i in range(40))
    assert len(split_looking_for(lf, max_sections=8)) == 8


def test_preamble_before_first_heading_is_dropped():
    lf = "loose text that belongs to no ask\n### Real\n" + "content " * 20
    secs = split_looking_for(lf, min_chars=5)
    assert len(secs) == 1 and secs[0].heading == "Real"


def test_empty_looking_for_uses_search_query():
    pair = {"userContactFile": {"lookingFor": None}, "searchQuery": "series A fintech investors"}
    secs = sections_for_pair(pair, min_chars=5)
    assert len(secs) == 1 and "fintech" in secs[0].text


# ------------------------------------------------------------------- pooling
@pytest.mark.parametrize("mode", POOLING_MODES)
def test_pooling_weights_are_a_distribution(mode):
    m = SectionedMoE(n_sim=3, emb_dim=8, pooling=mode, n_experts=2)
    m.eval()
    logits = torch.randn(7)
    gate_in = torch.randn(7, m.gate_proj.out_features)
    groups = [torch.tensor([0, 1, 2]), torch.tensor([3, 4]), torch.tensor([5, 6])]
    pooled, weights = m.pool(logits, gate_in, groups)
    assert pooled.shape == (3,)
    for w in weights:
        assert w.sum().item() == pytest.approx(1.0, abs=1e-5)
        assert (w >= 0).all()


def test_mean_pooling_is_the_plain_average():
    m = SectionedMoE(n_sim=3, emb_dim=8, pooling="mean", n_experts=2)
    m.eval()
    logits = torch.tensor([1.0, 2.0, 3.0, 10.0])
    gate_in = torch.zeros(4, m.gate_proj.out_features)
    pooled, _ = m.pool(logits, gate_in, [torch.tensor([0, 1, 2]), torch.tensor([3])])
    assert pooled[0].item() == pytest.approx(2.0)
    assert pooled[1].item() == pytest.approx(10.0)


def test_single_section_pair_passes_its_logit_through_unchanged():
    """A pair with one ask must not be reweighted by any pooling mode."""
    for mode in POOLING_MODES:
        m = SectionedMoE(n_sim=3, emb_dim=8, pooling=mode, n_experts=2)
        m.eval()
        logits = torch.tensor([4.2])
        pooled, _ = m.pool(logits, torch.randn(1, m.gate_proj.out_features), [torch.tensor([0])])
        assert pooled.item() == pytest.approx(4.2, abs=1e-5)


def test_softmax_pooling_leans_on_the_best_section():
    """Frozen-softmax control at tau=0.05 should be close to a max."""
    m = SectionedMoE(n_sim=3, emb_dim=8, pooling="softmax", pool_tau=0.05, n_experts=2)
    m.eval()
    logits = torch.tensor([0.0, 3.0, 0.5])
    pooled, w = m.pool(logits, torch.zeros(3, m.gate_proj.out_features), [torch.tensor([0, 1, 2])])
    assert w[0][1].item() > 0.99
    assert pooled.item() == pytest.approx(3.0, abs=0.05)


# --------------------------------------------------------------------- gate
def test_gate_routes_on_section_only():
    """The gate's input width is the section projection, not the expert input."""
    m = SectionedMoE(n_sim=3, emb_dim=64, gate_dims=16, n_experts=4)
    assert m.gate.in_features == 16
    assert m.gate_proj.in_features == 64
    # Expert input is similarity + projected interaction, and excludes the gate path.
    assert m.experts[0][0].in_features == 3 + m.interaction_proj.out_features


def test_expert_dropout_never_drops_every_expert():
    m = SectionedMoE(n_sim=3, emb_dim=8, n_experts=4, expert_dropout=0.99)
    m.train()
    torch.manual_seed(0)
    for _ in range(20):
        w = m.gate_weights(torch.randn(16, m.gate_proj.out_features))
        assert torch.isfinite(w).all()
        assert torch.allclose(w.sum(dim=-1), torch.ones(16), atol=1e-5)


def test_dropout_is_inactive_in_eval():
    m = SectionedMoE(n_sim=3, emb_dim=8, n_experts=4, expert_dropout=0.9)
    m.eval()
    x = torch.randn(5, m.gate_proj.out_features)
    assert torch.allclose(m.gate_weights(x), m.gate_weights(x))


def test_entropy_terms_pull_in_opposite_directions():
    """Sharpen wants each row decisive; balance wants the population spread."""
    decisive = torch.tensor([[0.98, 0.01, 0.01], [0.01, 0.98, 0.01]])
    uniform = torch.full((2, 3), 1.0 / 3)
    assert sharpen_loss(decisive) < sharpen_loss(uniform)

    collapsed = torch.tensor([[0.98, 0.01, 0.01], [0.98, 0.01, 0.01]])
    assert balance_loss(collapsed) > balance_loss(decisive)


def test_rejects_bad_hyperparameters():
    with pytest.raises(ValueError):
        SectionedMoE(n_sim=3, emb_dim=8, pooling="nonsense")
    with pytest.raises(ValueError):
        SectionedMoE(n_sim=3, emb_dim=8, tau=0.0)
    with pytest.raises(ValueError):
        SectionedMoE(n_sim=3, emb_dim=8, expert_dropout=1.0)


# ------------------------------------------------------------------ grouping
def test_regroup_renumbers_onto_sliced_rows():
    groups = [np.array([0, 1, 2]), np.array([3, 4]), np.array([5])]
    pair_idx = np.array([2, 0])
    rows = rows_for_pairs(groups, pair_idx)
    assert rows.tolist() == [5, 0, 1, 2]
    local = regroup(groups, pair_idx)
    assert [g.tolist() for g in local] == [[0], [1, 2, 3]]
    assert sum(len(g) for g in local) == len(rows)


def test_folds_keep_whole_seekers_together():
    seekers = [f"s{i // 3}" for i in range(30)]
    folds = seeker_disjoint_folds(seekers, 5, seed=0)
    seen: dict[str, int] = {}
    for f, idx in enumerate(folds):
        for i in idx:
            assert seen.setdefault(seekers[i], f) == f
    assert sum(len(f) for f in folds) == 30


# ------------------------------------------------------------------ encoders
def test_tfidf_backend_bypasses_the_broken_cache():
    """Same texts, different fits, must give different vectors.

    ``TfidfEncoder.encode()`` keys its cache without the fitted vocabulary, so
    going through it would return the first fit's vectors for both. That bug
    silently merged two arms of the previous experiment.
    """
    a = TfidfBackend().fit(["alpha beta gamma", "beta gamma delta"])
    b = TfidfBackend().fit(["zeta eta theta", "eta theta iota"])
    texts = ["alpha beta gamma"]
    assert not np.allclose(a.encode(texts), b.encode(texts))


def test_encoder_rows_are_unit_norm():
    e = TfidfBackend().fit(["alpha beta", "beta gamma", "gamma delta"])
    v = e.encode(["alpha beta", "gamma delta"])
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)


def test_mixed_embedding_spaces_are_rejected():
    class Fake:
        name = "qwen3"

    with pytest.raises(AssertionError, match="different backends"):
        assert_same_space(TfidfBackend(), Fake())


def test_text_hash_is_exact():
    assert text_hash("a") != text_hash("a ")


# --------------------------------------------------------- isolation guards
def test_quarantined_synth_stays_excluded():
    assert SectionedConfig().include_synth is False


@needs_data
def test_imported_pair_features_still_match_the_published_shape():
    """Drift guard on the read-only import from ``moe_rrf.features``.

    This package reuses that feature table rather than copying it. If its shape
    or column order changes, the pair-scalar block here silently means something
    different, so pin it.
    """
    from moe_rrf.features import FEATURE_NAMES

    assert len(FEATURE_NAMES) == 12
    assert FEATURE_NAMES[0] == "tfidf_cos"
    assert FEATURE_NAMES[1] == "tfidf_rank_pct"


@needs_data
def test_end_to_end_produces_one_score_per_pair():
    """The pipeline holds together: pairs in, sections exploded, pair scores out."""
    import numpy as np

    from moe_rrf.features import build_raw, tfidf_channel
    from moe_sectioned.data import load_real
    from moe_sectioned.features import RowStandardizer, build_section_features
    from moe_sectioned.train import fit, predict
    from baselines.bert_frozen.text import candidate_to_text

    pool, _ = load_real(DATA_DIR)
    sub = pool.rows[:24]
    y = pool.y[:24]
    seekers = pool.seeker_ids[:24]

    cos, rank = tfidf_channel(sub, sub)
    scalars = build_raw(sub, tfidf_cos=cos, tfidf_rank_pct=rank)
    enc = TfidfBackend().fit([candidate_to_text(p["matchContactFile"] or {}) for p in sub])

    feats = build_section_features(
        sub, list(y), seekers, encoder=enc,
        pair_scalars=scalars, pair_tfidf_cos=cos,
    )
    assert feats.n_rows > feats.n_pairs, "sections should multiply rows"
    assert feats.sim.shape == (feats.n_rows, 3)

    groups = feats.groups()
    assert sum(len(g) for g in groups) == feats.n_rows

    cfg = SectionedConfig(epochs=2)
    std = RowStandardizer().fit(feats.sim, feats.pair_scalars)
    X = std.transform(feats.sim, feats.pair_scalars)
    res = fit(
        cfg, sim=X, interaction=feats.interaction, section_emb=feats.section_emb,
        groups=groups, pair_labels=feats.pair_labels,
    )
    scores, gates, weights = predict(
        res.model, sim=X, interaction=feats.interaction,
        section_emb=feats.section_emb, groups=groups,
    )
    assert scores.shape == (feats.n_pairs,)
    assert gates.shape == (feats.n_rows, cfg.n_experts)
    assert len(weights) == feats.n_pairs
    assert np.isfinite(scores).all()
    assert ((scores >= 0) & (scores <= 1)).all()


# --------------------------------------------------- asymmetric encoding
def test_text_hash_is_role_qualified():
    """Qwen3 is asymmetric: the same string is a different vector per role.

    Before roles existed, sections and candidates were both encoded as documents
    and collided on one hash. Keying without the role would silently serve a
    document vector where a query vector was asked for.
    """
    assert text_hash("same text", "query") != text_hash("same text", "document")


def test_tfidf_ignores_role_because_it_is_symmetric():
    e = TfidfBackend().fit(["alpha beta", "beta gamma"])
    assert np.allclose(
        e.encode(["alpha beta"], role="query"),
        e.encode(["alpha beta"], role="document"),
    )


def test_qwen3_backend_refuses_partial_cache():
    """A missing vector must raise, never silently return a wrong row."""
    import json

    import pytest as _pytest

    from moe_sectioned.encode import Qwen3Backend

    tmp = Path(__file__).resolve().parent / "_tmp_emb"
    tmp.mkdir(exist_ok=True)
    try:
        np.save(tmp / "vectors.npy", np.eye(2, dtype=np.float32))
        (tmp / "index.json").write_text(
            json.dumps({"hash_to_row": {text_hash("known", "query"): 0}})
        )
        b = Qwen3Backend(embedding_dir=tmp)
        assert b.encode(["known"], role="query").shape == (1, 2)
        with _pytest.raises(KeyError, match="no cached Qwen3 vector"):
            b.encode(["unknown"], role="query")
    finally:
        for f in tmp.glob("*"):
            f.unlink()
        tmp.rmdir()


# ------------------------------------------------- embedding reduction
def test_reducer_cuts_parameters_by_orders_of_magnitude():
    """The fix for runs sec_001/sec_002, which were ~960k params on 708 rows."""
    from moe_sectioned.features import EmbeddingReducer

    rng = np.random.default_rng(0)
    sec = rng.normal(size=(60, 500)).astype(np.float32)
    inter = rng.normal(size=(60, 500)).astype(np.float32)
    r = EmbeddingReducer(n_components=16).fit(sec, inter)
    assert r.out_dims == 16
    assert r.transform(sec).shape == (60, 16)


def test_reducer_basis_comes_from_training_rows_only():
    """A basis fitted on evaluation rows would leak their structure."""
    from moe_sectioned.features import EmbeddingReducer

    rng = np.random.default_rng(1)
    train = rng.normal(size=(40, 30)).astype(np.float32)
    held = rng.normal(size=(40, 30)).astype(np.float32) + 50.0

    r = EmbeddingReducer(n_components=8).fit(train, train)
    before = r.basis.copy()
    r.transform(held)
    assert np.array_equal(r.basis, before), "transform must not refit"
