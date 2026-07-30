"""Tests for section-score aggregation modes.

The load-bearing test here is `test_record_score_is_batch_independent`. It pins
the invariant that a real bug violated: `noisy_or` originally rescaled scores by
the min/max of the matrix handed to it, and because
`eval_seeker._score_with_agg` aggregates the positive and negative matrices in
*separate* calls, the rescale became label-dependent and reported pair AUC
0.8500 against a ~0.60 field. Any mode that reaches for a statistic outside a
single record's own group of sections will fail that test.
"""

from __future__ import annotations

import numpy as np
import pytest

from baselines.voyage_nano_sectioned.aggregate import (
    AGG_FAMILY,
    AGG_MODES,
    aggregate_sections,
)


def _matrix(rng: np.random.Generator, n_sections: int, n_candidates: int) -> np.ndarray:
    """Cosine-like scores: float32, plausibly on [-1, 1]."""
    return rng.uniform(-0.2, 0.95, size=(n_sections, n_candidates)).astype(np.float32)


@pytest.mark.parametrize("mode", AGG_MODES)
def test_record_score_is_batch_independent(mode: str) -> None:
    """A record's aggregated score must not depend on which records share its batch.

    Aggregating one record alone must equal aggregating it inside a bigger
    batch. This is what makes it safe to call `aggregate_sections` separately on
    the positive and the negative matrices.
    """
    rng = np.random.default_rng(0)
    n_candidates = 7
    group_sizes = [1, 4, 2, 9, 3]
    matrix = _matrix(rng, sum(group_sizes), n_candidates)
    offsets = np.cumsum([0] + group_sizes).tolist()

    together = aggregate_sections(matrix, offsets, mode=mode)

    for i, size in enumerate(group_sizes):
        start = offsets[i]
        alone = aggregate_sections(
            matrix[start : start + size], [0, size], mode=mode
        )
        np.testing.assert_allclose(
            together[i],
            alone[0],
            rtol=1e-5,
            atol=1e-6,
            err_msg=(
                f"mode={mode!r}: record {i}'s score changed depending on batch "
                "composition, so it uses a statistic from outside its own group"
            ),
        )


@pytest.mark.parametrize("mode", AGG_MODES)
def test_output_shape_and_finiteness(mode: str) -> None:
    rng = np.random.default_rng(1)
    group_sizes = [3, 1, 6]
    matrix = _matrix(rng, sum(group_sizes), 5)
    offsets = np.cumsum([0] + group_sizes).tolist()

    out = aggregate_sections(matrix, offsets, mode=mode)

    assert out.shape == (len(group_sizes), 5)
    assert np.isfinite(out).all()


def test_relevance_and_veto_bracket_the_mean() -> None:
    """max >= mean >= min, and the soft modes sit inside their hard counterparts.

    This is the sanity property that makes the two families comparable: they are
    the same quantity read from opposite ends.
    """
    rng = np.random.default_rng(2)
    matrix = _matrix(rng, 12, 4)
    offsets = [0, 12]

    hi = aggregate_sections(matrix, offsets, mode="max")[0]
    mid = aggregate_sections(matrix, offsets, mode="mean")[0]
    lo = aggregate_sections(matrix, offsets, mode="min")[0]

    assert (hi >= mid - 1e-6).all()
    assert (mid >= lo - 1e-6).all()

    soft_hi = aggregate_sections(matrix, offsets, mode="softmax", temperature=0.05)[0]
    soft_lo = aggregate_sections(matrix, offsets, mode="softmin", temperature=0.05)[0]
    assert (soft_hi <= hi + 1e-6).all()
    assert (soft_lo >= lo - 1e-6).all()


def test_temperature_limits() -> None:
    """T -> 0 approaches the hard mode; large T approaches the mean."""
    rng = np.random.default_rng(3)
    matrix = _matrix(rng, 8, 3)
    offsets = [0, 8]

    hard_hi = aggregate_sections(matrix, offsets, mode="max")[0]
    hard_lo = aggregate_sections(matrix, offsets, mode="min")[0]
    plain = aggregate_sections(matrix, offsets, mode="mean")[0]

    np.testing.assert_allclose(
        aggregate_sections(matrix, offsets, mode="softmax", temperature=1e-4)[0],
        hard_hi,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        aggregate_sections(matrix, offsets, mode="softmin", temperature=1e-4)[0],
        hard_lo,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        aggregate_sections(matrix, offsets, mode="softmax", temperature=1e4)[0],
        plain,
        atol=1e-3,
    )


def test_noisy_or_is_length_normalized() -> None:
    """Duplicating a record's sections must not change its score.

    A raw product would shrink toward 0 with section count, making a seeker with
    126 asks unscorable regardless of fit. The geometric mean removes that.
    """
    rng = np.random.default_rng(4)
    group = _matrix(rng, 5, 3)

    once = aggregate_sections(group, [0, 5], mode="noisy_or")[0]
    twice = aggregate_sections(
        np.concatenate([group, group], axis=0), [0, 10], mode="noisy_or"
    )[0]

    np.testing.assert_allclose(once, twice, rtol=1e-5, atol=1e-6)


def test_noisy_or_rejects_a_dealbreaker() -> None:
    """One near-(-1) section must drag the score below an all-mediocre record.

    This is the property that distinguishes a veto from an average: a record
    that is excellent on four asks and catastrophic on one should score worse
    than a record that is uniformly mediocre.
    """
    excellent_but_one_dealbreaker = np.array(
        [[0.9], [0.9], [0.9], [0.9], [-0.95]], dtype=np.float32
    )
    uniformly_mediocre = np.full((5, 1), 0.35, dtype=np.float32)

    veto = aggregate_sections(excellent_but_one_dealbreaker, [0, 5], mode="noisy_or")[0, 0]
    plain = aggregate_sections(uniformly_mediocre, [0, 5], mode="noisy_or")[0, 0]
    assert veto < plain

    # ...whereas a plain average has the opposite ordering, which is exactly the
    # failure mode the veto family exists to avoid.
    avg_veto = aggregate_sections(excellent_but_one_dealbreaker, [0, 5], mode="mean")[0, 0]
    avg_plain = aggregate_sections(uniformly_mediocre, [0, 5], mode="mean")[0, 0]
    assert avg_veto > avg_plain


def test_unknown_mode_raises() -> None:
    matrix = np.zeros((3, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="unknown mode"):
        aggregate_sections(matrix, [0, 3], mode="not_a_mode")


def test_every_mode_has_a_family() -> None:
    assert set(AGG_FAMILY) == set(AGG_MODES)
    assert set(AGG_FAMILY.values()) == {"relevance", "veto", "average"}
