"""Unit tests for reciprocal_lambda_grid_ask_offer — the same no-fitting
lambda sweep as the other reciprocal_lambda_grid* packages, scored with
ask_offer_001's two independently-trained LoRA towers (Ask + Offer) instead
of one shared encoder.
"""

from __future__ import annotations

import numpy as np

from baselines.reciprocal_static.eval import build_lambda_grid
from baselines.reciprocal_static.text import bg_text as static_bg_text
from baselines.reciprocal_static.text import look_text as static_look_text
from baselines.reciprocal_static.text import seeker_look_text as static_seeker_look_text
from reciprocal_lambda_grid_ask_offer.eval import sweep_lambda

FULL_PROFILE = {
    "positioning": "Product lead at a seed-stage climate startup.",
    "background": "Ten years in hardware, two prior exits.",
    "lookingFor": "Investors focused on climate tech.",
    "notes": "Met at a conference last year.",
}


def test_text_builders_reused_read_only_from_reciprocal_static() -> None:
    """This package imports baselines.reciprocal_static.text directly rather
    than duplicating it (same as twotower_ask_offer.eval does) — pin that the
    functions used are the exact ones, not silently-forked copies."""
    from reciprocal_lambda_grid_ask_offer import eval as sweep_eval

    assert sweep_eval.bg_text is static_bg_text
    assert sweep_eval.look_text is static_look_text
    assert sweep_eval.seeker_look_text is static_seeker_look_text


def test_sweep_lambda_returns_one_point_per_grid_value_no_selection() -> None:
    grid = build_lambda_grid(-1.0, 1.0, 0.5)
    labels = np.array([1, 1, 0, 0], dtype=np.int32)
    s_fwd = np.array([0.9, 0.8, 0.1, 0.2])
    s_recip = np.array([0.1, -0.1, 0.2, -0.2])
    curve = sweep_lambda(s_fwd, s_recip, labels, grid)
    assert len(curve) == len(grid)
    assert [pt["lambda"] for pt in curve] == list(grid)


def test_sweep_lambda_at_zero_matches_forward_only_auc() -> None:
    from sklearn.metrics import roc_auc_score

    grid = np.array([0.0])
    labels = np.array([1, 1, 0, 0], dtype=np.int32)
    s_fwd = np.array([0.9, 0.8, 0.1, 0.2])
    s_recip = np.array([0.5, -0.3, 0.4, -0.1])
    curve = sweep_lambda(s_fwd, s_recip, labels, grid)
    assert curve[0]["pair_auc"] == roc_auc_score(labels, s_fwd)
