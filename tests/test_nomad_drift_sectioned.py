"""CPU-only checks for nomad_drift_sectioned — no GPU/model needed.

The load-bearing claims this file guards:

* the rrf_003 loader carries a ground-truth ``true_section_index`` for every
  record, matching the staged pair's own ``metadata.section_index`` exactly —
  without this, selection accuracy is checked against nothing.
* section selection picks the *closest* section by cosine to the query, and
  correctly falls back to a seeker's only section (or whole profile) when
  there is nothing to select between.
* ``_reconstruct_raw_by_pair_index`` undoes ``encode_everything``'s pos/neg
  split back into the original pair_ids order exactly — a silent misalignment
  here would blend every seeker's section vector with the *wrong* pair's query.
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


def test_loader_section_index_matches_staged_metadata() -> None:
    from nomad_drift_sectioned.calibrate import load_sectioned_records

    records = load_sectioned_records(RRF_BATCH)
    manifest = json.loads((RRF_BATCH / "manifest.json").read_text())
    assert len(records) == len(manifest["records"])

    by_key = {r.pair_key: r for r in records}
    # spot-check a sample against the staged file directly
    for rec in manifest["records"][:20]:
        staged = json.loads((RRF_BATCH / rec["path"]).read_text())
        assert by_key[rec["pair_key"]].true_section_index == staged["metadata"]["section_index"]


def test_select_and_align_picks_closest_section_and_scores_accuracy() -> None:
    from nomad_drift_sectioned.calibrate import SectionEncoded, SectionedRrfRecord, select_and_align

    d = 4
    e = np.eye(d, dtype=np.float32)

    records = [
        SectionedRrfRecord(
            pair_key="p1", seeker_id="s1", candidate_id="c1", query_key="q1", label="pos",
            user_contact_file={}, match_contact_file={}, search_query="", true_section_index=0,
        ),
        SectionedRrfRecord(
            pair_key="p2", seeker_id="s1", candidate_id="c2", query_key="q2", label="neg",
            user_contact_file={}, match_contact_file={}, search_query="", true_section_index=1,
        ),
    ]
    enc = SectionEncoded(
        records=records,
        query_by_key={"q1": e[0], "q2": e[1]},  # q1 matches section 0, q2 matches section 1
        candidate_by_id={"c1": e[2], "c2": e[3]},
        sections_by_seeker={"s1": [(0, e[0]), (1, e[1])]},  # exact match for both queries
    )

    arrays, stats = select_and_align(enc)
    assert stats["n_checkable"] == 2
    assert stats["n_correct"] == 2
    assert stats["selection_accuracy"] == 1.0
    np.testing.assert_allclose(arrays.profile[0], e[0])
    np.testing.assert_allclose(arrays.profile[1], e[1])


def test_select_and_align_single_section_is_not_checkable() -> None:
    """A seeker with only one section has nothing to select between."""
    from nomad_drift_sectioned.calibrate import SectionEncoded, SectionedRrfRecord, select_and_align

    d = 3
    e = np.eye(d, dtype=np.float32)
    records = [
        SectionedRrfRecord(
            pair_key="p1", seeker_id="s1", candidate_id="c1", query_key="q1", label="pos",
            user_contact_file={}, match_contact_file={}, search_query="", true_section_index=0,
        ),
    ]
    enc = SectionEncoded(
        records=records,
        query_by_key={"q1": e[0]},
        candidate_by_id={"c1": e[1]},
        sections_by_seeker={"s1": [(-1, e[2])]},  # WHOLE sentinel, single entry
    )
    arrays, stats = select_and_align(enc)
    assert stats["n_checkable"] == 0
    assert stats["selection_accuracy"] is None
    np.testing.assert_allclose(arrays.profile[0], e[2])


def test_reconstruct_raw_by_pair_index_undoes_the_split() -> None:
    from nomad_drift_sectioned.report import _reconstruct_raw_by_pair_index

    class FakeEncoded:
        pair_ids = ["a", "b", "c", "d"]
        positives = [{"id": "b"}, {"id": "d"}]
        negatives = [{"id": "a"}, {"id": "c"}]
        pos_index = [1, 3]
        neg_index = [0, 2]

    raw = _reconstruct_raw_by_pair_index(FakeEncoded())
    assert [r["id"] for r in raw] == ["a", "b", "c", "d"]
