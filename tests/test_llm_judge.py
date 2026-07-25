"""Tests for baselines.llm_judge — no LLM, no network."""

from __future__ import annotations

import numpy as np
import pytest

from baselines.llm_judge.judge import (
    VerdictCache,
    judge_all,
    parse_verdict,
    prompt_hash,
    verdict_to_score,
)
from baselines.llm_judge.metrics import calibration_buckets, decision_metrics
from baselines.llm_judge.prompt import SYSTEM_PROMPTS, build_user_prompt

PROFILE_A = {
    "positioning": "Founder of a seed-stage fintech.",
    "lookingFor": "Intros to pre-seed investors.",
    "notes": "",
}
PROFILE_B = {
    "positioning": "Partner at a pre-seed fund.",
    "background": "Ten years investing in fintech.",
}


# --------------------------------------------------------------------------
# The load-bearing invariant of the whole experiment
# --------------------------------------------------------------------------


def test_search_query_never_reaches_the_prompt():
    """The query must not appear in the user prompt or any system prompt.

    This is the experiment's premise, not an incidental detail — a regression
    here would silently turn it into a different (and much easier) experiment.
    """
    query = "US-based pre-seed fintech investors who move quickly"
    a = dict(PROFILE_A, searchQuery=query)
    user = build_user_prompt(a, PROFILE_B)
    assert query not in user
    assert "searchQuery" not in user
    for system in SYSTEM_PROMPTS.values():
        assert query not in system


def test_build_user_prompt_includes_both_profiles():
    user = build_user_prompt(PROFILE_A, PROFILE_B)
    assert "PERSON A" in user and "PERSON B" in user
    assert "seed-stage fintech" in user
    assert "Partner at a pre-seed fund" in user
    # Empty fields are dropped by profile_to_text, not rendered as "notes: ".
    assert "notes:" not in user


def test_max_field_truncates_and_none_does_not():
    long_profile = {"positioning": "x" * 5000}
    assert len(build_user_prompt(long_profile, PROFILE_B, max_field=100)) < 1000
    assert "x" * 5000 in build_user_prompt(long_profile, PROFILE_B, max_field=None)


# --------------------------------------------------------------------------
# Verdict parsing / scoring
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected_match,expected_conf",
    [
        ({"match": "yes", "confidence": 80}, "yes", 80.0),
        ({"match": "YES", "confidence": 80}, "yes", 80.0),  # case
        ({"match": " no ", "confidence": 80}, "no", 80.0),  # whitespace
        ({"match": True, "confidence": 80}, "yes", 80.0),  # JSON bool
        ({"match": False, "confidence": 80}, "no", 80.0),
        ({"match": "yes", "confidence": "85%"}, "yes", 85.0),  # string pct
        ({"match": "yes", "confidence": 0.9}, "yes", 90.0),  # 0-1 scale
        ({"match": "yes", "confidence": 1.0}, "yes", 1.0),  # integral: stays as-is
    ],
)
def test_parse_verdict_accepts_common_model_variations(raw, expected_match, expected_conf):
    v = parse_verdict(raw)
    assert v["match"] == expected_match
    assert v["confidence"] == pytest.approx(expected_conf)


@pytest.mark.parametrize(
    "raw",
    [
        {"match": "maybe", "confidence": 50},
        {"match": "yes"},
        {"confidence": 50},
        {"match": "yes", "confidence": 150},
        {"match": "yes", "confidence": -1},
        {"match": None, "confidence": 50},
    ],
)
def test_parse_verdict_rejects_unusable_output(raw):
    """Malformed answers must raise so judge_pair retries instead of coercing."""
    with pytest.raises(ValueError):
        parse_verdict(raw)


def test_verdict_to_score_is_centered_on_the_decision_boundary():
    assert verdict_to_score({"match": "yes", "confidence": 100}) == 1.0
    assert verdict_to_score({"match": "no", "confidence": 100}) == 0.0
    # A 0-confidence answer of either sign lands exactly on 0.5, so
    # accuracy_at_0.5 tracks the model's own decision.
    assert verdict_to_score({"match": "yes", "confidence": 0}) == 0.5
    assert verdict_to_score({"match": "no", "confidence": 0}) == 0.5
    # Monotone in belief.
    assert verdict_to_score({"match": "yes", "confidence": 90}) > verdict_to_score(
        {"match": "yes", "confidence": 60}
    )
    assert verdict_to_score({"match": "no", "confidence": 60}) > verdict_to_score(
        {"match": "no", "confidence": 90}
    )


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------


def test_prompt_hash_changes_with_prompt_text():
    """Editing a prompt must invalidate cached verdicts."""
    assert prompt_hash("sys", "user") == prompt_hash("sys", "user")
    assert prompt_hash("sys", "user") != prompt_hash("sys", "user!")
    assert prompt_hash("sys", "user") != prompt_hash("sys!", "user")
    # Field boundary: the two halves must not be concatenation-ambiguous.
    assert prompt_hash("ab", "c") != prompt_hash("a", "bc")


def test_cache_round_trips_and_prevents_repeat_calls(tmp_path, monkeypatch):
    import baselines.llm_judge.judge as judge_mod

    calls: list[str] = []

    def fake(*, system, user, model, temperature, api_key, base_url, run_tags=None):
        calls.append(user)
        return {"match": "yes", "confidence": 70, "reasoning": "r"}

    monkeypatch.setattr(judge_mod, "complete_json", fake)

    requests = [("k1", "sys", "u1"), ("k2", "sys", "u2")]
    kwargs = dict(
        model="m", temperature=0.0, api_key="k", base_url="http://x", workers=2
    )

    cache = VerdictCache(tmp_path / "v.json")
    verdicts, errors = judge_all(requests, cache=cache, **kwargs)
    assert not errors
    assert set(verdicts) == {"k1", "k2"}
    assert len(calls) == 2

    # Fresh cache object reading the same file: no further API calls.
    reloaded = VerdictCache(tmp_path / "v.json")
    assert len(reloaded) == 2
    verdicts2, errors2 = judge_all(requests, cache=reloaded, **kwargs)
    assert not errors2
    assert len(calls) == 2, "cache hit still called the API"
    assert verdicts2["k1"]["match"] == "yes"


def test_judge_all_reports_failures_without_aborting(tmp_path, monkeypatch):
    import baselines.llm_judge.judge as judge_mod

    def fake(*, system, user, model, temperature, api_key, base_url, run_tags=None):
        if user == "bad":
            raise RuntimeError("upstream 500")
        return {"match": "no", "confidence": 60}

    monkeypatch.setattr(judge_mod, "complete_json", fake)
    monkeypatch.setattr(judge_mod.time, "sleep", lambda _s: None)

    verdicts, errors = judge_all(
        [("ok", "sys", "good"), ("bad", "sys", "bad")],
        model="m",
        temperature=0.0,
        api_key="k",
        base_url="http://x",
        cache=VerdictCache(tmp_path / "v.json"),
        workers=2,
        max_attempts=2,
    )
    assert set(verdicts) == {"ok"}
    assert "bad" in errors and "upstream 500" in errors["bad"]
    # A failure must not be cached, or a re-run could never retry it.
    assert set(VerdictCache(tmp_path / "v.json")._data) == {"ok"}


# --------------------------------------------------------------------------
# Decision metrics
# --------------------------------------------------------------------------


def _v(match: str, conf: float = 80.0) -> dict:
    return {"match": match, "confidence": conf}


def test_decision_metrics_perfect_judge():
    m = decision_metrics([_v("yes")] * 4, [_v("no")] * 4)
    assert m["accuracy"] == 1.0
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["f1"] == 1.0
    assert m["confusion"] == {
        "true_positive": 4,
        "false_positive": 0,
        "true_negative": 4,
        "false_negative": 0,
    }
    assert m["yes_rate"] == 0.5


def test_decision_metrics_flags_an_always_yes_judge():
    """Perfect recall with zero information — yes_rate is what exposes it."""
    m = decision_metrics([_v("yes")] * 5, [_v("yes")] * 5)
    assert m["recall"] == 1.0
    assert m["accuracy"] == 0.5
    assert m["yes_rate"] == 1.0
    assert m["yes_rate_on_negatives"] == 1.0


def test_decision_metrics_confidence_split():
    m = decision_metrics([_v("yes", 90), _v("no", 50)], [_v("no", 70), _v("no", 70)])
    # correct: pos#1 (90), neg#1 (70), neg#2 (70); wrong: pos#2 (50)
    assert m["mean_confidence_when_correct"] == pytest.approx((90 + 70 + 70) / 3)
    assert m["mean_confidence_when_wrong"] == pytest.approx(50.0)


def test_calibration_buckets_partition_all_verdicts():
    pos = [_v("yes", c) for c in (55, 65, 75, 85, 95, 100)]
    neg = [_v("no", c) for c in (0, 59.9, 60)]
    buckets = calibration_buckets(pos, neg)
    assert sum(b["n"] for b in buckets) == len(pos) + len(neg)
    # 100 must land in the top bucket, not fall off the end.
    assert buckets[-1]["n"] == 2  # 95 and 100


def test_decision_metrics_matches_pair_accuracy_at_half():
    """decision.accuracy and pair.accuracy_at_0.5 must agree by construction."""
    from baselines.metrics import pair_metrics

    pos = [_v("yes", 90), _v("no", 60), _v("yes", 55)]
    neg = [_v("no", 80), _v("yes", 70), _v("no", 65)]
    d = decision_metrics(pos, neg)
    p = pair_metrics(
        np.array([verdict_to_score(v) for v in pos]),
        np.array([verdict_to_score(v) for v in neg]),
    )
    assert d["accuracy"] == pytest.approx(p["accuracy_at_0.5"])
