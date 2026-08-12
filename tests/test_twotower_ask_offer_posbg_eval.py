"""Tests for the pos+bg offer-text swap. No model download."""

from __future__ import annotations

from twotower_ask_offer_posbg_eval.text import BG_FIELDS, bg_text, look_text, seeker_look_text

FULL = {
    "positioning": "POS",
    "background": "BG",
    "lookingFor": "LF",
    "notes": "NOTES",
    "locationAvailability": "LOC",
    "introPreferences": "INTRO",
    "personalPreferences": "PERS",
    "meetingAndSchedulingPreferences": "MEET",
}


def test_offer_fields_are_positioning_and_background_only():
    assert BG_FIELDS == ("positioning", "background")
    text = bg_text(FULL)
    assert "positioning: POS" in text
    assert "background: BG" in text
    for banned in (
        "lookingFor",
        "NOTES",
        "LOC",
        "INTRO",
        "PERS",
        "MEET",
        "Search query",
    ):
        assert banned not in text


def test_offer_text_narrower_than_reciprocal_static():
    from baselines.reciprocal_static.text import bg_text as wide_bg

    assert "notes: NOTES" in wide_bg(FULL)
    assert "notes: NOTES" not in bg_text(FULL)


def test_ask_text_matches_reciprocal_static():
    from baselines.reciprocal_static.text import look_text as shared_look
    from baselines.reciprocal_static.text import seeker_look_text as shared_seeker

    assert look_text(FULL) == shared_look(FULL)
    assert seeker_look_text(FULL, "QUERY") == shared_seeker(FULL, "QUERY")
    assert "Search query: QUERY" in seeker_look_text(FULL, "QUERY")


def test_eval_has_no_holdout_loop():
    import inspect

    from twotower_ask_offer_posbg_eval import eval as ev

    src = inspect.getsource(ev.run_eval)
    assert "holdout_pairs" not in src
    assert "real_holdout" not in src
    assert 'subset="all"' in src or "subset='all'" in src
