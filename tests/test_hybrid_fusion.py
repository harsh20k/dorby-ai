"""Unit tests for TF-IDF + Voyage late-fusion helpers (no model load)."""

from __future__ import annotations

import numpy as np
import pytest

from baselines.hybrid_tfidf_voyage.fusion import (
    fit_alpha_fusion,
    fit_logistic_fusion,
    rrf_score_matrix,
)
from baselines.metrics import (
    ranks_from_score_matrix,
    retrieval_metrics_from_ranks,
    retrieval_metrics_from_score_matrix,
)


def test_alpha_fusion_prefers_informative_channel():
    # TF-IDF perfectly separates; Voyage is noise → α should land near 1.
    rng = np.random.default_rng(0)
    y = np.array([1] * 20 + [0] * 20)
    tfidf = np.where(y == 1, 0.8, 0.2) + rng.normal(0, 0.01, size=40)
    voyage = rng.normal(0.5, 0.05, size=40)
    model = fit_alpha_fusion(tfidf, voyage, y)
    # Perfect separation is reached by α≈0.65 on this draw; any α that
    # weights TF-IDF more than Voyage is the qualitative claim.
    assert model.alpha > 0.5
    assert model.fit_auc > 0.95


def test_logistic_fusion_learns_positive_tfidf_coef():
    rng = np.random.default_rng(1)
    y = np.array([1] * 30 + [0] * 30)
    tfidf = np.where(y == 1, 0.7, 0.3) + rng.normal(0, 0.02, size=60)
    voyage = rng.normal(0.5, 0.1, size=60)
    model = fit_logistic_fusion(tfidf, voyage, y)
    assert model.coef_tfidf > 0
    scores = model.pair_scores(tfidf, voyage)
    assert scores[y == 1].mean() > scores[y == 0].mean()
    assert model.fit_auc > 0.9


def test_rrf_prefers_shared_top_ranks():
    # Candidate 0 is rank-1 for both → should win RRF.
    tfidf = np.array([[0.9, 0.5, 0.1]], dtype=np.float64)
    voyage = np.array([[0.8, 0.4, 0.2]], dtype=np.float64)
    fused = rrf_score_matrix(tfidf, voyage, k=60)
    assert fused.shape == (1, 3)
    assert int(np.argmax(fused[0])) == 0


def test_retrieval_from_score_matrix_matches_ranks():
    scores = np.array(
        [
            [0.1, 0.9, 0.5],  # target idx 1 → rank 1
            [0.8, 0.2, 0.7],  # target idx 0 → rank 1
        ],
        dtype=np.float64,
    )
    target_ids = ["b", "a"]
    candidate_ids = ["a", "b", "c"]
    ranks = ranks_from_score_matrix(scores, target_ids, candidate_ids)
    assert ranks == [1, 1]
    metrics = retrieval_metrics_from_score_matrix(scores, target_ids, candidate_ids)
    assert metrics["mrr"] == pytest.approx(1.0)
    assert metrics["recall@1"] == pytest.approx(1.0)
    assert retrieval_metrics_from_ranks(ranks)["mrr"] == pytest.approx(1.0)
