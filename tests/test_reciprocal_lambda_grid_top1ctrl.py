"""Unit tests for baselines/reciprocal_lambda_grid_top1ctrl — the same
no-fitting lambda sweep as baselines/reciprocal_lambda_grid, but scored with
the fine-tuned top1_ctrl LoRA adapter instead of frozen voyage-4-nano.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from baselines.reciprocal_lambda_grid.text import bg_text as frozen_bg_text
from baselines.reciprocal_lambda_grid.text import look_text as frozen_look_text
from baselines.reciprocal_lambda_grid.text import seeker_look_text as frozen_seeker_look_text
from baselines.reciprocal_lambda_grid_top1ctrl.encode import TOP1_CTRL_ADAPTER_DIR, cosine_scores
from baselines.reciprocal_lambda_grid_top1ctrl.eval import sweep_lambda
from baselines.reciprocal_lambda_grid_top1ctrl.text import bg_text, look_text, seeker_look_text
from baselines.reciprocal_static.eval import build_lambda_grid

REPO = Path(__file__).resolve().parent.parent

FULL_PROFILE = {
    "positioning": "Product lead at a seed-stage climate startup.",
    "background": "Ten years in hardware, two prior exits.",
    "lookingFor": "Investors focused on climate tech.",
    "notes": "Met at a conference last year.",
}


def test_bg_text_matches_frozen_field_choice() -> None:
    """Same field set as the frozen-model sweep — pin numeric parity."""
    assert bg_text(FULL_PROFILE) == frozen_bg_text(FULL_PROFILE)


def test_look_text_matches_frozen_exactly() -> None:
    assert look_text(FULL_PROFILE) == frozen_look_text(FULL_PROFILE)


def test_seeker_look_text_matches_frozen_exactly() -> None:
    assert seeker_look_text(FULL_PROFILE, "climate investors") == frozen_seeker_look_text(
        FULL_PROFILE, "climate investors"
    )


def test_cosine_scores_matches_dot_product_for_unit_vectors() -> None:
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    b = np.array([[1.0, 0.0], [0.7071, 0.7071]])
    scores = cosine_scores(a, b)
    assert scores[0] == 1.0
    assert abs(scores[1] - 0.7071) < 1e-3


def test_sweep_lambda_returns_one_point_per_grid_value_no_selection() -> None:
    grid = build_lambda_grid(-1.0, 1.0, 0.5)
    labels = np.array([1, 1, 0, 0], dtype=np.int32)
    s_fwd = np.array([0.9, 0.8, 0.1, 0.2])
    s_recip = np.array([0.1, -0.1, 0.2, -0.2])
    curve = sweep_lambda(s_fwd, s_recip, labels, grid)
    assert len(curve) == len(grid)
    assert [pt["lambda"] for pt in curve] == list(grid)


def test_top1_ctrl_adapter_dir_exists() -> None:
    if not (REPO / TOP1_CTRL_ADAPTER_DIR).exists():
        import pytest

        pytest.skip("top1_ctrl adapter not present in this checkout")
    assert (REPO / TOP1_CTRL_ADAPTER_DIR / "adapter_config.json").exists()
