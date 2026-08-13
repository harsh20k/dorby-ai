"""Unit tests for reciprocal_lambda_grid_ask_offer_posbg — the same no-fitting
lambda sweep as reciprocal_lambda_grid_ask_offer, scored with
ask_offer_posbg_001's two independently-trained LoRA towers, offer text =
positioning + background only (not the wide field set).
"""

from __future__ import annotations

import numpy as np

from baselines.reciprocal_static.eval import build_lambda_grid
from baselines.reciprocal_static.text import look_text as static_look_text
from baselines.reciprocal_static.text import seeker_look_text as static_seeker_look_text
from twotower_ask_offer_posbg.text import BG_FIELDS
from twotower_ask_offer_posbg.text import bg_text as posbg_bg_text
from twotower_ask_offer_posbg.text import look_text as posbg_look_text
from twotower_ask_offer_posbg.text import seeker_look_text as posbg_seeker_look_text
from reciprocal_lambda_grid_ask_offer_posbg.eval import sweep_lambda

FULL_PROFILE = {
    "positioning": "Product lead at a seed-stage climate startup.",
    "background": "Ten years in hardware, two prior exits.",
    "lookingFor": "Investors focused on climate tech.",
    "notes": "Met at a conference last year.",
    "locationAvailability": "SF Bay Area, remote OK.",
}


def test_text_builders_reused_read_only_from_posbg_package() -> None:
    """This package imports twotower_ask_offer_posbg.text directly rather
    than the wide baselines.reciprocal_static.text used by the original
    ask_offer sweep — pin that the functions used are the exact ones."""
    from reciprocal_lambda_grid_ask_offer_posbg import eval as sweep_eval

    assert sweep_eval.bg_text is posbg_bg_text
    assert sweep_eval.look_text is posbg_look_text
    assert sweep_eval.seeker_look_text is posbg_seeker_look_text


def test_offer_fields_are_positioning_and_background_only() -> None:
    assert BG_FIELDS == ("positioning", "background")
    text = posbg_bg_text(FULL_PROFILE)
    assert "positioning:" in text
    assert "background:" in text
    assert "notes" not in text.lower()
    assert "locationAvailability" not in text
    assert "lookingFor" not in text


def test_ask_side_matches_reciprocal_static() -> None:
    """Ask packing is unchanged vs the original ask_offer sweep."""
    assert posbg_look_text is not static_look_text  # different modules
    assert posbg_look_text(FULL_PROFILE) == static_look_text(FULL_PROFILE)
    assert posbg_seeker_look_text(FULL_PROFILE, "climate investors") == static_seeker_look_text(
        FULL_PROFILE, "climate investors"
    )


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
