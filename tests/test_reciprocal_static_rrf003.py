"""Unit tests for baselines/reciprocal_static_rrf003 — lambda calibrated on
rrf_003 judge labels instead of real train pairs, background narrowed to
positioning + background only.

Deliberate duplicate of tests/test_reciprocal_static.py's coverage shape
(experiment-isolation rule: the two packages must be pinned separately so
they can't silently drift), plus parity checks against the original package
for the two functions that are meant to behave identically (look_text,
seeker_look_text) and a divergence check for the one that was deliberately
narrowed (bg_text).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from baselines.reciprocal_static.eval import build_lambda_grid, fit_lambda
from baselines.reciprocal_static.text import bg_text as original_bg_text
from baselines.reciprocal_static.text import look_text as original_look_text
from baselines.reciprocal_static.text import seeker_look_text as original_seeker_look_text
from baselines.reciprocal_static_rrf003.eval import build_bg_corpus
from baselines.reciprocal_static_rrf003.rrf003_data import DEFAULT_RRF003_DIR, load_rrf003_pairs
from baselines.reciprocal_static_rrf003.text import bg_text, look_text, seeker_look_text
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
        source="synth",
    )


# ---------------------------------------------------------------------------
# text split — narrowed bg_text
# ---------------------------------------------------------------------------


def test_bg_text_is_positioning_and_background_only() -> None:
    text = bg_text(FULL_PROFILE)
    assert "positioning" in text
    assert "background" in text
    for field in ("lookingFor", "notes", "locationAvailability", "introPreferences",
                  "personalPreferences", "meetingAndSchedulingPreferences"):
        assert field not in text


def test_bg_text_diverges_from_original_wider_field_set() -> None:
    """Deliberate divergence: original includes notes/preferences fields, this doesn't."""
    narrowed = bg_text(FULL_PROFILE)
    original = original_bg_text(FULL_PROFILE)
    assert narrowed != original
    assert "Met at a conference last year" in original
    assert "Met at a conference last year" not in narrowed


def test_look_text_matches_original_exactly() -> None:
    """look_text is unchanged from the original package — pin numeric parity."""
    assert look_text(FULL_PROFILE) == original_look_text(FULL_PROFILE)


def test_seeker_look_text_matches_original_exactly() -> None:
    assert seeker_look_text(FULL_PROFILE, "climate investors") == original_seeker_look_text(
        FULL_PROFILE, "climate investors"
    )


# ---------------------------------------------------------------------------
# background corpus dedup (duplicate of the original's build_bg_corpus test,
# using the narrowed text.bg_text via this package's own build_bg_corpus)
# ---------------------------------------------------------------------------


def test_build_bg_corpus_dedupes_by_match_contact_id() -> None:
    pairs = [_pair(match_contact_id="cmcand1"), _pair(match_contact_id="cmcand1"), _pair(match_contact_id="cmcand2")]
    ids, texts = build_bg_corpus(pairs)
    assert ids == ["cmcand1", "cmcand2"]
    assert len(texts) == 2
    assert "lookingFor" not in texts[0]


# ---------------------------------------------------------------------------
# lambda fitting is reused unchanged from baselines.reciprocal_static.eval —
# just a smoke test that the import path works and behaves as expected.
# ---------------------------------------------------------------------------


def test_fit_lambda_recovers_informative_reciprocal_signal() -> None:
    rng = np.random.default_rng(1)
    n = 200
    labels = np.array([1] * (n // 2) + [0] * (n // 2), dtype=np.int32)
    s_fwd = rng.normal(0.0, 1.0, n)
    s_recip = np.where(labels == 1, 1.0, -1.0) + rng.normal(0, 0.05, n)
    grid = build_lambda_grid(-2.0, 2.0, 0.05)
    lam, auc = fit_lambda(s_fwd, s_recip, labels, grid)
    assert auc > 0.9
    assert lam != 0.0


# ---------------------------------------------------------------------------
# rrf_003 loader sanity — protects run_eval's fitting population against
# silent drift of the checked-in export.
# ---------------------------------------------------------------------------


def test_rrf003_loader_matches_manifest_counts() -> None:
    if not DEFAULT_RRF003_DIR.exists():
        pytest.skip("exports/rrf_datasets/rrf_003 not present")
    rrf003 = load_rrf003_pairs()
    assert rrf003.batch_id == "rrf_003"
    assert rrf003.n_pos == 1175
    assert rrf003.n_neg == 1444
    assert len(rrf003.pairs) == 1175 + 1444
    assert all(p.source == "synth" for p in rrf003.pairs)
