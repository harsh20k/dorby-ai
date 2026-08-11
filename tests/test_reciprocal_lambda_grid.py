"""Unit tests for baselines/reciprocal_lambda_grid — a lambda sensitivity
sweep (no fitting step) over the static reciprocal score, bg_text narrowed
to positioning + background only.
"""

from __future__ import annotations

import numpy as np

from baselines.reciprocal_lambda_grid.eval import sweep_lambda
from baselines.reciprocal_lambda_grid.text import bg_text, look_text, seeker_look_text
from baselines.reciprocal_static.eval import build_lambda_grid
from baselines.reciprocal_static.text import look_text as original_look_text
from baselines.reciprocal_static.text import seeker_look_text as original_seeker_look_text
from baselines.reciprocal_static_rrf003.text import bg_text as rrf003_bg_text

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


def test_bg_text_is_positioning_and_background_only() -> None:
    text = bg_text(FULL_PROFILE)
    assert "positioning" in text
    assert "background" in text
    for field in ("lookingFor", "notes", "locationAvailability"):
        assert field not in text


def test_bg_text_matches_reciprocal_static_rrf003_field_choice() -> None:
    """Same field choice as the rrf003 experiment — pin numeric parity."""
    assert bg_text(FULL_PROFILE) == rrf003_bg_text(FULL_PROFILE)


def test_look_text_matches_original_exactly() -> None:
    assert look_text(FULL_PROFILE) == original_look_text(FULL_PROFILE)


def test_seeker_look_text_matches_original_exactly() -> None:
    assert seeker_look_text(FULL_PROFILE, "climate investors") == original_seeker_look_text(
        FULL_PROFILE, "climate investors"
    )


def test_sweep_lambda_returns_one_point_per_grid_value_no_selection() -> None:
    """The sweep must not pick a winner — it reports every grid point."""
    grid = build_lambda_grid(-1.0, 1.0, 0.5)
    labels = np.array([1, 1, 0, 0], dtype=np.int32)
    s_fwd = np.array([0.9, 0.8, 0.1, 0.2])
    s_recip = np.array([0.1, -0.1, 0.2, -0.2])
    curve = sweep_lambda(s_fwd, s_recip, labels, grid)
    assert len(curve) == len(grid)
    assert [pt["lambda"] for pt in curve] == list(grid)
    assert all(0.0 <= pt["pair_auc"] <= 1.0 for pt in curve)


def test_sweep_lambda_at_zero_matches_forward_only_auc() -> None:
    from sklearn.metrics import roc_auc_score

    grid = build_lambda_grid(-1.0, 1.0, 1.0)  # includes 0.0
    labels = np.array([1, 1, 0, 0], dtype=np.int32)
    s_fwd = np.array([0.9, 0.8, 0.1, 0.2])
    s_recip = np.array([0.5, -0.5, 0.9, -0.9])  # would change ranking if not ignored at lambda=0
    curve = sweep_lambda(s_fwd, s_recip, labels, grid)
    zero_point = next(pt for pt in curve if pt["lambda"] == 0.0)
    assert zero_point["pair_auc"] == roc_auc_score(labels, s_fwd)
