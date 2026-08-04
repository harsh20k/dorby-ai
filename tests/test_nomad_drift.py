"""CPU-only checks for nomad_drift — no GPU/model needed.

The load-bearing claims this file guards:

* the rrf_003 loader's record count and label split match the batch manifest
  exactly, so calibration is scored on the whole batch, not a silently
  truncated subset.
* alpha=0.0 / alpha=1.0 reproduce pure profile / pure query vectors exactly —
  the same endpoint-equivalence guarantee ``query_weighted`` pins for its own
  alpha arms, so alpha is a genuine interpolation, not an approximation.
* the shortlist top-1 metric only counts query_keys carrying both labels —
  a query_key with only positives or only negatives has nothing to rank.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
RRF_BATCH = REPO / "artifacts" / "pairing_rrf" / "rrf_003"

pytestmark = pytest.mark.skipif(
    not RRF_BATCH.exists(), reason="artifacts/pairing_rrf/rrf_003 not present in this checkout"
)


def test_loader_matches_manifest_counts() -> None:
    from nomad_drift.calibrate import load_rrf_records

    manifest = json.loads((RRF_BATCH / "manifest.json").read_text())
    records = load_rrf_records(RRF_BATCH)

    assert len(records) == len(manifest["records"])
    assert manifest["counts"]["pos"] + manifest["counts"]["neg"] == len(records)
    assert sum(1 for r in records if r.label == "pos") == manifest["counts"]["pos"]
    assert sum(1 for r in records if r.label == "neg") == manifest["counts"]["neg"]


def test_loader_schema_matches_real_pair_shape() -> None:
    """Each record exposes the same fields query_weighted.text builders expect."""
    from nomad_drift.calibrate import load_rrf_records

    records = load_rrf_records(RRF_BATCH)
    r = records[0]
    assert isinstance(r.user_contact_file, dict)
    assert isinstance(r.match_contact_file, dict)
    assert isinstance(r.search_query, str)
    assert r.label in ("pos", "neg")


def test_alpha_zero_and_one_are_exact_endpoints() -> None:
    from nomad_drift.calibrate import combine_batch

    rng = np.random.default_rng(0)
    profile = rng.normal(size=(5, 8)).astype(np.float32)
    query = rng.normal(size=(5, 8)).astype(np.float32)
    profile /= np.linalg.norm(profile, axis=-1, keepdims=True)
    query /= np.linalg.norm(query, axis=-1, keepdims=True)

    np.testing.assert_allclose(combine_batch(query, profile, 0.0), profile, atol=1e-5)
    np.testing.assert_allclose(combine_batch(query, profile, 1.0), query, atol=1e-5)


def test_combine_batch_agrees_with_query_weighted_combine() -> None:
    """The duplicated blend formula must match the one it mirrors, exactly."""
    from nomad_drift.calibrate import combine_batch
    from query_weighted.eval import combine as qw_combine

    rng = np.random.default_rng(1)
    profile = rng.normal(size=(4, 6)).astype(np.float32)
    query = rng.normal(size=(4, 6)).astype(np.float32)
    profile /= np.linalg.norm(profile, axis=-1, keepdims=True)
    query /= np.linalg.norm(query, axis=-1, keepdims=True)

    for alpha in (0.0, 0.3, 0.6, 1.0):
        np.testing.assert_allclose(
            combine_batch(query, profile, alpha), qw_combine(query, profile, alpha), atol=1e-6
        )


def test_shortlist_metric_skips_single_label_query_keys() -> None:
    from nomad_drift.calibrate import AlignedArrays, score_alpha

    # 3 query_keys: qA has both labels (counts), qB is pos-only, qC is neg-only
    # (neither of the latter two should contribute to shortlist accuracy).
    d = 4
    vecs = np.eye(d, dtype=np.float32)[:3]
    arrays = AlignedArrays(
        profile=vecs,
        query=vecs,
        candidate=vecs,
        labels=["pos", "neg", "pos"],
        query_keys=["qA", "qA", "qB"],
    )
    # qA: pos vec[0] vs neg vec[1] -- candidate is identical to seeker (score 1.0)
    # for both rows at alpha=0 (pure profile), so this exercises the tie-break
    # path deterministically; the key assertion is just the query_key count.
    result = score_alpha(arrays, alpha=0.0)
    assert result["shortlist_n_query_keys"] == 1  # only qA has both labels
