"""Tests for twotower_ask_offer_posbg: pure tensor-math correctness of the
reciprocal loss (no model/GPU needed), the text-reuse pin against
baselines.reciprocal_static, frozen-rows schema, and config sanity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

torch.manual_seed(0)


def test_offer_text_is_positioning_and_background_only():
    """This package's whole point vs twotower_ask_offer: offer = pos+bg."""
    from twotower_ask_offer_posbg.loss import bg_text, look_text, seeker_look_text
    from twotower_ask_offer_posbg.text import BG_FIELDS, bg_text as text_bg

    assert BG_FIELDS == ("positioning", "background")
    assert bg_text is text_bg
    profile = {
        "positioning": "POS",
        "background": "BG",
        "lookingFor": "LF",
        "notes": "NOTES",
    }
    offer = bg_text(profile)
    assert "POS" in offer and "BG" in offer
    assert "NOTES" not in offer and "LF" not in offer
    assert "LF" in look_text(profile)
    assert "QUERY" in seeker_look_text(profile, "QUERY")


class TestCombineAndCrossEntropy:
    def test_shapes_and_labels(self):
        from twotower_ask_offer_posbg.loss import combine_and_cross_entropy

        n, p, d = 3, 6, 8
        k_seek = torch.randn(n, d)
        v_seek = torch.randn(n, d)
        k_pool = torch.randn(p, d)
        v_pool = torch.randn(p, d)

        loss, s = combine_and_cross_entropy(k_seek, v_seek, k_pool, v_pool, lam=1.0, scale=20.0)
        assert s.shape == (n, p)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_matches_manual_formula(self):
        from twotower_ask_offer_posbg.loss import combine_and_cross_entropy

        n, p, d = 2, 4, 3
        k_seek = torch.randn(n, d)
        v_seek = torch.randn(n, d)
        k_pool = torch.randn(p, d)
        v_pool = torch.randn(p, d)
        lam = 0.7

        _, s = combine_and_cross_entropy(k_seek, v_seek, k_pool, v_pool, lam=lam, scale=20.0)

        s_fwd_manual = k_seek @ v_pool.T
        s_rev_manual = v_seek @ k_pool.T
        expected = s_fwd_manual + lam * s_rev_manual
        assert torch.allclose(s, expected, atol=1e-6)

    def test_perfect_pool_gives_near_zero_loss(self):
        """If pool item i is an exact copy of seeker i's ask/offer target
        (and orthogonal to everything else), cross-entropy should push
        toward ~0 as scale grows — a sanity check the loss actually rewards
        the true diagonal, not some other pattern."""
        from twotower_ask_offer_posbg.loss import combine_and_cross_entropy

        n, d = 4, 16
        k_seek = torch.eye(n, d)
        v_seek = torch.eye(n, d)
        k_pool = torch.eye(n, d)
        v_pool = torch.eye(n, d)

        loss, s = combine_and_cross_entropy(k_seek, v_seek, k_pool, v_pool, lam=1.0, scale=50.0)
        assert torch.argmax(s, dim=1).tolist() == list(range(n))
        assert loss.item() < 0.1

    def test_raises_on_pool_smaller_than_batch(self):
        from twotower_ask_offer_posbg.loss import combine_and_cross_entropy

        k_seek = torch.randn(5, 4)
        v_seek = torch.randn(5, 4)
        k_pool = torch.randn(3, 4)
        v_pool = torch.randn(3, 4)
        with pytest.raises(ValueError):
            combine_and_cross_entropy(k_seek, v_seek, k_pool, v_pool, lam=1.0, scale=20.0)

    def test_raises_on_ask_offer_shape_mismatch(self):
        from twotower_ask_offer_posbg.loss import combine_and_cross_entropy

        k_seek = torch.randn(3, 4)
        v_seek = torch.randn(3, 5)  # mismatched dim
        k_pool = torch.randn(6, 4)
        v_pool = torch.randn(6, 4)
        with pytest.raises(ValueError):
            combine_and_cross_entropy(k_seek, v_seek, k_pool, v_pool, lam=1.0, scale=20.0)


class TestBuildBatchTexts:
    def _row(self, seeker_id, positive_id, negative_ids):
        from twotower_ask_offer_posbg.data import AskOfferRow

        profile = {
            "positioning": "does robots",
            "background": "phd robotics",
            "lookingFor": "series a investors",
        }
        return AskOfferRow(
            query_key=f"{seeker_id}::q0",
            seeker_id=seeker_id,
            positive_id=positive_id,
            negative_ids=tuple(negative_ids),
            search_query="deep tech investors",
            seeker_profile=profile,
            positive_profile=profile,
            negative_profiles=tuple(profile for _ in negative_ids),
        )

    def test_pool_ordering_positives_then_negatives(self):
        from twotower_ask_offer_posbg.loss import build_batch_texts

        rows = [self._row(f"s{i}", f"p{i}", [f"n{i}"]) for i in range(3)]
        bt = build_batch_texts(rows)

        assert bt.n == 3
        assert len(bt.seeker_ask) == 3
        assert len(bt.pool_ask) == 6  # 3 positives + 3 negatives
        assert len(bt.pool_offer) == 6
        # search query only appears on the seeker side
        assert "deep tech investors" in bt.seeker_ask[0]
        assert all("deep tech investors" not in t for t in bt.pool_ask)

    def test_two_negatives_per_anchor_appends_second_block(self):
        from twotower_ask_offer_posbg.loss import build_batch_texts

        rows = [self._row(f"s{i}", f"p{i}", [f"n{i}a", f"n{i}b"]) for i in range(2)]
        bt = build_batch_texts(rows)
        assert len(bt.pool_ask) == 6  # 2 positives + 2*2 negatives


class TestFrozenRowsSchema:
    def test_frozen_rows_load_if_present(self):
        """If the frozen rows file exists (import_rows.py has been run),
        confirm its schema matches what data.load_rows expects. Skips if
        absent — the file is gitignored (artifacts/), not committed."""
        path = Path("artifacts/twotower_ask_offer_posbg/ask_offer_rows.json")
        if not path.exists():
            pytest.skip("frozen rows not built locally (run: python -m twotower_ask_offer_posbg.import_rows)")

        from twotower_ask_offer_posbg.data import load_rows

        rows, provenance = load_rows(path)
        assert len(rows) > 0
        assert provenance["source_rows_sha256"]
        row = rows[0]
        assert row.seeker_id
        assert row.positive_id
        assert isinstance(row.negative_profiles, tuple)
        assert "lookingFor" in row.seeker_profile or "positioning" in row.seeker_profile


class TestConfig:
    def test_effective_batch_matches_target(self):
        from twotower_ask_offer_posbg.config import EFFECTIVE_BATCH_TARGET, build_config

        cfg = build_config(run_id="test_run")
        eff = cfg.tower.train_batch_size * cfg.tower.gradient_accumulation_steps
        assert eff == EFFECTIVE_BATCH_TARGET

    def test_run_dir_includes_run_id(self):
        from twotower_ask_offer_posbg.config import build_config

        cfg = build_config(run_id="my_run", output_dir=Path("artifacts/twotower_ask_offer_posbg"))
        assert cfg.run_dir == Path("artifacts/twotower_ask_offer_posbg/my_run")

    def test_default_lambda_matches_reciprocal_static_fit(self):
        from twotower_ask_offer_posbg.config import AskOfferConfig

        assert AskOfferConfig(run_id="x").lam == 1.75
