"""Unit tests for baselines/reciprocal_static — the static half of Ga Wu's
"Dynamic Reciprocal User Matching with Fast Weight Programmers" paper
(no FWP memory, since this repo has no per-user interaction logs; see the
module docstring in baselines/reciprocal_static/eval.py for the scope call).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from baselines.reciprocal_static.eval import build_bg_corpus, build_lambda_grid, fit_lambda
from baselines.reciprocal_static.text import bg_text, look_text, seeker_look_text
from eval_real_full.data import load_real_pairs
from twotower.data import LabeledPair

REPO = Path(__file__).resolve().parent.parent

FULL_PROFILE = {
    "positioning": "Product lead at a seed-stage climate startup.",
    "background": "Ten years in hardware, two prior exits.",
    "lookingFor": "Investors focused on climate tech.",
    "notes": "Met at a conference last year.",
    "locationAvailability": "SF, available weekday evenings.",
    "introPreferences": "Warm intros only.",
    "personalPreferences": "Prefers async.",
    "meetingAndSchedulingPreferences": "30-minute calls.",
}


def _pair(match_contact_id: str = "cmcand1", user_contact_id: str = "cmseeker1") -> LabeledPair:
    return LabeledPair(
        pair_id=f"pos:{user_contact_id}:{match_contact_id}",
        label="pos",
        pair={
            "userContactId": user_contact_id,
            "matchContactId": match_contact_id,
            "searchQuery": "climate investors",
            "userContactFile": FULL_PROFILE,
            "matchContactFile": FULL_PROFILE,
        },
        source="real_train",
    )


# ---------------------------------------------------------------------------
# text split
# ---------------------------------------------------------------------------


def test_bg_text_excludes_looking_for() -> None:
    text = bg_text(FULL_PROFILE)
    assert "lookingFor" not in text
    assert "Investors focused on climate tech" not in text
    for field in ("positioning", "background", "notes"):
        assert field in text


def test_look_text_is_looking_for_only() -> None:
    text = look_text(FULL_PROFILE)
    assert text == "lookingFor: Investors focused on climate tech."


def test_look_text_empty_when_field_missing() -> None:
    assert look_text({"positioning": "x"}) == ""


def test_seeker_look_text_appends_query() -> None:
    text = seeker_look_text(FULL_PROFILE, "climate investors")
    assert "lookingFor: Investors focused on climate tech." in text
    assert "Search query: climate investors" in text


def test_seeker_look_text_no_query_falls_back_to_profile_only() -> None:
    text = seeker_look_text(FULL_PROFILE, "")
    assert text == look_text(FULL_PROFILE)


def test_bg_and_look_text_are_disjoint_field_coverage() -> None:
    """Every non-lookingFor field lands in bg_text, lookingFor lands only in look_text."""
    bg = bg_text(FULL_PROFILE)
    look = look_text(FULL_PROFILE)
    assert "Investors focused on climate tech" in look
    assert "Investors focused on climate tech" not in bg


# ---------------------------------------------------------------------------
# lambda fitting
# ---------------------------------------------------------------------------


def test_fit_lambda_never_beaten_by_forward_only_floor() -> None:
    rng = np.random.default_rng(0)
    n = 200
    labels = np.array([1] * (n // 2) + [0] * (n // 2), dtype=np.int32)
    s_fwd = np.where(labels == 1, rng.normal(0.6, 0.3, n), rng.normal(0.4, 0.3, n))
    s_recip = rng.normal(0.0, 1.0, n)  # pure noise, uncorrelated with label
    grid = build_lambda_grid(-2.0, 2.0, 0.1)
    lam, auc = fit_lambda(s_fwd, s_recip, labels, grid)
    from sklearn.metrics import roc_auc_score

    forward_only_auc = roc_auc_score(labels, s_fwd)
    assert auc >= forward_only_auc - 1e-9


def test_fit_lambda_recovers_informative_reciprocal_signal() -> None:
    """When s_recip alone perfectly separates the label and s_fwd is noise,
    grid search should find a lambda that lifts AUC well above the forward-only floor."""
    rng = np.random.default_rng(1)
    n = 200
    labels = np.array([1] * (n // 2) + [0] * (n // 2), dtype=np.int32)
    s_fwd = rng.normal(0.0, 1.0, n)  # noise
    s_recip = np.where(labels == 1, 1.0, -1.0) + rng.normal(0, 0.05, n)
    grid = build_lambda_grid(-2.0, 2.0, 0.05)
    lam, auc = fit_lambda(s_fwd, s_recip, labels, grid)
    assert auc > 0.9
    assert lam != 0.0


def test_fit_lambda_zero_is_a_valid_outcome() -> None:
    """lambda=0 (pure forward-only) must always be in the search space's reach,
    i.e. included by construction as the floor even if not in the grid."""
    grid = build_lambda_grid(0.5, 2.0, 0.5)  # deliberately excludes 0.0
    assert 0.0 not in grid
    labels = np.array([1, 1, 0, 0], dtype=np.int32)
    s_fwd = np.array([0.9, 0.8, 0.1, 0.2])
    s_recip = np.array([0.0, 0.0, 0.0, 0.0])  # zero vector: any lambda gives identical score
    lam, auc = fit_lambda(s_fwd, s_recip, labels, grid)
    assert auc == pytest.approx(1.0)


def test_build_lambda_grid_bounds_and_step() -> None:
    grid = build_lambda_grid(-1.0, 1.0, 0.5)
    assert grid[0] == -1.0
    assert grid[-1] == 1.0
    assert len(grid) == 5


# ---------------------------------------------------------------------------
# background corpus dedup
# ---------------------------------------------------------------------------


def test_build_bg_corpus_dedupes_by_match_contact_id() -> None:
    pairs = [_pair(match_contact_id="cmcand1"), _pair(match_contact_id="cmcand1"), _pair(match_contact_id="cmcand2")]
    ids, texts = build_bg_corpus(pairs)
    assert ids == ["cmcand1", "cmcand2"]
    assert len(texts) == 2


# ---------------------------------------------------------------------------
# real-data population sanity (no model loading — protects run_eval's
# train/holdout partition assumptions against silent data drift)
# ---------------------------------------------------------------------------


def test_real_pair_population_matches_frozen_split() -> None:
    data_dir = REPO / "data"
    split_path = data_dir / "synthetic" / "seed_split.json"
    if not data_dir.exists():
        pytest.skip("data/ not present (e.g. running from a worktree without the data checkout)")
    real_all = load_real_pairs(data_dir, split_path, subset="all")
    assert real_all.n_pos == 100
    assert real_all.n_neg == 100
    assert real_all.n_candidates == 178

    train = load_real_pairs(data_dir, split_path, subset="train")
    holdout = load_real_pairs(data_dir, split_path, subset="holdout")
    assert len(train.pairs) == 131
    assert len(holdout.pairs) == 69
    assert len(train.pairs) + len(holdout.pairs) == len(real_all.pairs)
