"""Tests for the MMoE re-ranker.

The load-bearing ones are the gate-regularization tests: they pin that the two
entropy terms actually pull in opposite directions, which is the property the
slides' `L_sharp` alone does not have (sharpening by itself is minimized by
collapsing every example onto one expert).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from moe_reranker import diagnostics
from moe_reranker.features import FEATURE_NAMES, FeatureBuilder
from moe_reranker.model import MMoE, balance_loss, sharpen_loss, task_loss


# ------------------------------------------------------------------- model
def test_forward_shapes_and_gate_is_a_distribution() -> None:
    m = MMoE(n_features=14, n_experts=3, n_tasks=2)
    x = torch.randn(11, 14)
    logits, gates = m(x)
    assert logits.shape == (11, 2)
    assert gates.shape == (11, 2, 3)
    np.testing.assert_allclose(
        gates.detach().numpy().sum(axis=-1), 1.0, rtol=1e-5, atol=1e-6
    )


def test_lower_tau_polarizes_the_gate() -> None:
    """This is the professor's Idea 1: temperature sharpens routing."""
    x = torch.randn(64, 14)
    ents = []
    for tau in (1.0, 0.2, 0.05):
        torch.manual_seed(0)
        m = MMoE(n_features=14, n_experts=3, tau=tau, expert_dropout=0.0)
        m.eval()
        with torch.no_grad():
            _, g = m(x)
        ents.append(float(sharpen_loss(g)))
    assert ents[0] > ents[1] > ents[2], f"entropy should fall as tau falls, got {ents}"


def test_expert_dropout_only_active_in_training() -> None:
    m = MMoE(n_features=6, n_experts=4, expert_dropout=0.9, tau=1.0)
    x = torch.randn(32, 6)
    m.eval()
    with torch.no_grad():
        a, _ = m(x)
        b, _ = m(x)
    torch.testing.assert_close(a, b)  # deterministic when not training

    m.train()
    torch.manual_seed(0)
    _, g = m(x)
    # Even at dropout 0.9, no row may lose every expert.
    assert (g.detach().numpy().sum(axis=-1) > 0.99).all()


def test_rejects_bad_hyperparameters() -> None:
    with pytest.raises(ValueError):
        MMoE(n_features=4, tau=0.0)
    with pytest.raises(ValueError):
        MMoE(n_features=4, expert_dropout=1.0)


# ------------------------------------------------------- gate regularization
def test_sharpen_and_balance_oppose_each_other() -> None:
    """A collapsed gate minimizes sharpening but maximizes balance loss.

    This is why both terms are needed: optimizing `sharpen_loss` alone is solved
    by sending every example to one expert, which `balance_loss` penalizes.
    """
    n, t, m = 40, 1, 3
    collapsed = np.zeros((n, t, m), dtype=np.float32)
    collapsed[:, :, 0] = 1.0
    collapsed = torch.from_numpy(np.clip(collapsed, 1e-7, 1.0))

    # Sharp per example, but every example picks a different expert.
    spread = np.full((n, t, m), 1e-7, dtype=np.float32)
    for i in range(n):
        spread[i, :, i % m] = 1.0
    spread = torch.from_numpy(spread)

    uniform = torch.full((n, t, m), 1.0 / m)

    # Both collapsed and spread are maximally sharp...
    assert float(sharpen_loss(collapsed)) < float(sharpen_loss(uniform))
    assert float(sharpen_loss(spread)) < float(sharpen_loss(uniform))
    # ...but only `spread` keeps the experts balanced.
    assert float(balance_loss(spread)) < float(balance_loss(collapsed))


def test_balance_loss_is_minimized_by_uniform_usage() -> None:
    n, t, m = 30, 1, 3
    uniform = torch.full((n, t, m), 1.0 / m)
    skewed = torch.from_numpy(
        np.tile(np.array([[[0.8, 0.1, 0.1]]], dtype=np.float32), (n, t, 1))
    )
    assert float(balance_loss(uniform)) < float(balance_loss(skewed))
    np.testing.assert_allclose(float(balance_loss(uniform)), -np.log(m), atol=1e-5)


def test_task_loss_ignores_masked_targets() -> None:
    """A pair with no auxiliary label must not be trained as if the label were 0."""
    logits = torch.zeros(4, 2)
    targets = torch.tensor(
        [[1.0, 1.0], [0.0, 1.0], [1.0, 1.0], [0.0, 1.0]], dtype=torch.float32
    )
    all_on = torch.ones(4, 2, dtype=torch.bool)
    aux_off = torch.tensor([[True, False]] * 4)

    with_aux = float(task_loss(logits, targets, all_on, (1.0, 1.0)))
    without = float(task_loss(logits, targets, aux_off, (1.0, 1.0)))
    assert without < with_aux
    # Zero weight on the auxiliary task must equal masking it out entirely.
    np.testing.assert_allclose(
        float(task_loss(logits, targets, all_on, (1.0, 0.0))), without, atol=1e-6
    )


# ------------------------------------------------------------- diagnostics
def _gates(assign: list[int], n_experts: int = 3, n_tasks: int = 1) -> np.ndarray:
    g = np.full((len(assign), n_tasks, n_experts), 1e-6, dtype=np.float64)
    for i, a in enumerate(assign):
        g[i, :, a] = 1.0
    return g / g.sum(axis=-1, keepdims=True)


def test_diagnostics_flags_collapse() -> None:
    d = diagnostics.compute(_gates([0] * 30), [f"s{i}" for i in range(30)])
    assert any("COLLAPSE" in w for w in d.warnings)
    assert d.collapse_ratio[0] > 0.99


def test_diagnostics_flags_a_non_routing_gate() -> None:
    uniform = np.full((30, 1, 3), 1.0 / 3)
    d = diagnostics.compute(uniform, [f"s{i}" for i in range(30)])
    assert any("not routing" in w for w in d.warnings)


def test_diagnostics_clean_when_balanced_and_decisive() -> None:
    """Balanced usage + one-hot routing that is independent of the seeker.

    Note the trap this test originally fell into: ``assign = i % 3`` alongside
    ``seekers = s{i % 6}`` makes routing a perfect function of the seeker, which
    Diagnostic 3 correctly flagged. The routing has to be drawn independently.
    """
    rng = np.random.default_rng(7)
    seekers = [f"s{i % 6}" for i in range(90)]  # 15 pairs per seeker
    assign = list(rng.integers(0, 3, 90))
    d = diagnostics.compute(_gates(assign), seekers, n_permutations=200)
    assert d.warnings == [], d.warnings


def test_routing_null_is_reported_and_beats_raw_mi_for_interpretation() -> None:
    """With one pair per seeker, random routing already scores high raw MI.

    The whole point of the permutation null: the raw number is near its ceiling
    by construction, so `excess` must stay near zero for random routing.
    """
    rng = np.random.default_rng(0)
    assign = list(rng.integers(0, 3, 100))
    seekers = [f"s{i}" for i in range(100)]  # every seeker unique
    d = diagnostics.compute(_gates(assign), seekers, n_permutations=200)

    assert d.seeker_routing_mi_normalized[0] > 0.5  # raw MI looks alarming...
    assert abs(d.seeker_routing_excess[0]) < 0.15  # ...but excess says otherwise
    assert not any("tracks seeker identity" in w for w in d.warnings)


def test_diagnostics_detects_genuine_seeker_routing() -> None:
    """Routing that really is a function of the seeker must be caught."""
    seekers = [f"s{i % 5}" for i in range(100)]
    assign = [int(s[1:]) % 3 for s in seekers]  # routing determined by seeker
    d = diagnostics.compute(_gates(assign), seekers, n_permutations=200)
    assert any("tracks seeker identity" in w for w in d.warnings), d.warnings


def test_diagnostics_shape_validation() -> None:
    with pytest.raises(ValueError, match="n_tasks"):
        diagnostics.compute(np.zeros((5, 3)), ["a"] * 5)
    with pytest.raises(ValueError, match="seeker_ids"):
        diagnostics.compute(np.full((5, 1, 3), 1 / 3), ["a"] * 4)


# ---------------------------------------------------------------- features
def test_feature_builder_refuses_holdout_and_requires_fit() -> None:
    b = FeatureBuilder()
    raw = np.random.default_rng(0).normal(size=(10, len(FEATURE_NAMES)))
    with pytest.raises(ValueError, match="fit on train only"):
        b.fit(raw, is_holdout=True)
    with pytest.raises(RuntimeError, match="fit"):
        FeatureBuilder().transform(raw)


def test_standardization_uses_train_statistics_only() -> None:
    rng = np.random.default_rng(0)
    train = rng.normal(size=(50, len(FEATURE_NAMES)))
    other = rng.normal(loc=100.0, size=(10, len(FEATURE_NAMES)))
    b = FeatureBuilder().fit(train)

    zt = b.transform(train)
    np.testing.assert_allclose(zt.mean(axis=0), 0.0, atol=1e-5)
    # The shifted split must come out shifted, not re-centered on itself.
    assert b.transform(other).mean() > 10.0


def test_constant_feature_does_not_explode() -> None:
    raw = np.random.default_rng(0).normal(size=(20, len(FEATURE_NAMES)))
    raw[:, 3] = 7.0
    z = FeatureBuilder().fit(raw).transform(raw)
    assert np.isfinite(z).all()
    np.testing.assert_allclose(z[:, 3], 0.0, atol=1e-5)
