"""Pin the query-weighting arms against the baselines they must stay comparable to.

The load-bearing claims this file guards:

* ``concat_baseline`` is byte-identical to what every published baseline encodes.
  Without that the whole experiment has no anchor.
* ``profile_only`` is byte-identical to the existing ``*_no_query`` ablation.
* the α combination is a true interpolation whose endpoints reproduce the two
  text arms exactly, so α is a single interpretable knob.
* the easy/hard negative split does not move between arms — otherwise arms are
  scored on different populations and hard-neg AUC cannot be compared.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# text builders
# --------------------------------------------------------------------------

SAMPLE_PROFILE = {
    "positioning": "Founder of a climate fintech.",
    "background": "Ten years in carbon markets.",
    "lookingFor": "### Capital\nSeries A leads.",
    "notes": "Prefers warm intros.",
}
SAMPLE_QUERY = "Looking for climate-focused Series A investors in Europe."


def test_concat_baseline_matches_the_published_seeker_text() -> None:
    from baselines.bert_frozen.text import seeker_to_text

    from query_weighted.text import concat_baseline

    assert concat_baseline(SAMPLE_PROFILE, SAMPLE_QUERY) == seeker_to_text(
        SAMPLE_PROFILE, SAMPLE_QUERY
    )


def test_profile_only_matches_the_no_query_ablation() -> None:
    from baselines.text_no_query import seeker_to_text as no_query_seeker

    from query_weighted.text import profile_only

    assert profile_only(SAMPLE_PROFILE, SAMPLE_QUERY) == no_query_seeker(SAMPLE_PROFILE)


def test_query_prefix_agrees_with_the_original() -> None:
    """The duplicated 'Search query: ' prefix must still match seeker_to_text."""
    from query_weighted.text import concat_baseline, profile_only, query_block

    full = concat_baseline(SAMPLE_PROFILE, SAMPLE_QUERY)
    body = profile_only(SAMPLE_PROFILE, SAMPLE_QUERY)
    assert full == f"{body}\n\n{query_block(SAMPLE_QUERY)}"


def test_query_first_reorders_without_changing_content() -> None:
    from query_weighted.text import concat_baseline, query_first

    a = concat_baseline(SAMPLE_PROFILE, SAMPLE_QUERY)
    b = query_first(SAMPLE_PROFILE, SAMPLE_QUERY)
    assert a != b
    assert sorted(a.split()) == sorted(b.split()), "front-loading must not add tokens"


def test_repeat_one_equals_query_first() -> None:
    from query_weighted.text import query_first, query_repeated_front

    assert query_repeated_front(SAMPLE_PROFILE, SAMPLE_QUERY, repeats=1) == query_first(
        SAMPLE_PROFILE, SAMPLE_QUERY
    )


def test_repeats_scale_the_query_share() -> None:
    from query_weighted.text import query_block, query_repeated_front

    q = query_block(SAMPLE_QUERY)
    for k in (1, 3, 5):
        assert query_repeated_front(SAMPLE_PROFILE, SAMPLE_QUERY, repeats=k).count(q) == k


def test_empty_query_degrades_to_profile() -> None:
    """A pair with no searchQuery must never encode an empty string."""
    from query_weighted.text import profile_only, query_first, query_only, query_repeated_front

    body = profile_only(SAMPLE_PROFILE, "")
    assert query_only(SAMPLE_PROFILE, "") == body
    assert query_first(SAMPLE_PROFILE, "") == body
    assert query_repeated_front(SAMPLE_PROFILE, "", repeats=5) == body


def test_repeats_must_be_positive() -> None:
    from query_weighted.text import query_repeated_front

    with pytest.raises(ValueError):
        query_repeated_front(SAMPLE_PROFILE, SAMPLE_QUERY, repeats=0)


# --------------------------------------------------------------------------
# vector combination
# --------------------------------------------------------------------------


def test_combine_endpoints_reproduce_the_text_arms() -> None:
    from query_weighted.eval import combine

    rng = np.random.default_rng(0)
    q = rng.normal(size=(7, 16)).astype(np.float32)
    p = rng.normal(size=(7, 16)).astype(np.float32)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    p /= np.linalg.norm(p, axis=-1, keepdims=True)

    assert np.allclose(combine(q, p, 1.0), q, atol=1e-6)
    assert np.allclose(combine(q, p, 0.0), p, atol=1e-6)


def test_combine_returns_unit_vectors() -> None:
    from query_weighted.eval import combine

    rng = np.random.default_rng(1)
    q = rng.normal(size=(5, 8)).astype(np.float32)
    p = rng.normal(size=(5, 8)).astype(np.float32)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    p /= np.linalg.norm(p, axis=-1, keepdims=True)

    for alpha in (0.1, 0.5, 0.9):
        norms = np.linalg.norm(combine(q, p, alpha), axis=-1)
        assert np.allclose(norms, 1.0, atol=1e-5)


# --------------------------------------------------------------------------
# end-to-end plumbing, no model required
# --------------------------------------------------------------------------


class _FakeEncoder:
    """Deterministic hash-based unit vectors — exercises the pipeline, not a model."""

    model_name = "fake"
    max_length = 128
    truncate_dim = 32

    def encode(self, texts, *, role, batch_size=4, show_progress=False):
        out = np.zeros((len(texts), 32), dtype=np.float32)
        for i, t in enumerate(texts):
            digest = hashlib.sha256(f"{role}:{t}".encode()).digest()
            vec = np.frombuffer(digest * 4, dtype=np.uint8)[:32].astype(np.float32)
            vec = vec - vec.mean()
            out[i] = vec / max(float(np.linalg.norm(vec)), 1e-12)
        return out


def _has_data() -> bool:
    return (REPO / "data" / "dataset_positive.json").is_file() and (
        REPO / "eval_real_full" / "data_frozen" / "real_200_manifest.json"
    ).is_file()


@pytest.mark.skipif(not _has_data(), reason="data/ or the frozen manifest is absent")
def test_pipeline_shapes_and_fixed_hardness_split() -> None:
    """Every arm must see the same populations; only the seeker vectors differ."""
    from query_weighted.eval import run_all_arms

    result = run_all_arms(
        _FakeEncoder(),
        REPO / "data",
        REPO / "data" / "synthetic" / "seed_split.json",
        subsets=("all", "holdout"),
        alphas=(0.5,),
    )

    arms = result["arms"]
    assert "concat_baseline" in arms and "alpha_0.5" in arms

    for subset, n_pairs, n_cand in (("all", 200, 178), ("holdout", 69, 65)):
        counts = {
            (a["n_pos"], a["n_neg"], a["n_candidates"])
            for arm in arms.values()
            for s, a in arm.items()
            if s == subset
        }
        assert counts == {(n_pairs // 2 if subset == "all" else 29,
                           n_pairs // 2 if subset == "all" else 40,
                           n_cand)}, f"{subset} population differs between arms: {counts}"

    # The hard/easy negative split must be identical across arms, since it is
    # defined from the pinned baseline text rather than each arm's own text.
    sizes = {
        arm_name: (
            arm["all"]["slices"]["neg_hardness"]["hard"]["n_negatives"],
            arm["all"]["slices"]["neg_hardness"]["easy"]["n_negatives"],
            arm["all"]["slices"]["neg_hardness"]["hard"]["overlap_cutoff"],
        )
        for arm_name, arm in arms.items()
    }
    assert len(set(sizes.values())) == 1, f"hardness split moved between arms: {sizes}"
