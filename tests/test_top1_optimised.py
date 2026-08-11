"""Pin twotower_top1_optimised's copy and its two load-bearing changes.

The isolation rule requires copying rather than editing a prior experiment's code
and pinning each copy so the two cannot drift. `data.py` is a verbatim copy of
the ablation's; if it drifts, this arm stops training on the population Arm A
trained on and the comparison silently stops meaning anything.

The other tests guard the two changes the arm exists to make — a run that
silently fell back to library-default loss settings, or to triplet-accuracy
checkpoint selection, would look like a valid result while testing nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def test_data_py_matches_ablation() -> None:
    copy = REPO / "twotower_top1_optimised" / "data.py"
    orig = REPO / "twotower_rrf_triplet_ablation" / "data.py"
    normalised = copy.read_text(encoding="utf-8").replace(
        "twotower_top1_optimised", "twotower_rrf_triplet_ablation"
    )
    assert normalised == orig.read_text(encoding="utf-8"), (
        "data.py has drifted from the ablation's copy — this arm would no longer "
        "train on the same rows as Arm A, breaking the comparison."
    )


def test_preset_selects_on_recall_at_1() -> None:
    """Change 2: checkpoint selection must optimise recall@1, not triplet accuracy."""
    from twotower_top1_optimised.config import build_config

    cfg = build_config(
        "voyage-4-nano",
        run_id="unit_test",
        rows_path=Path("/tmp/rows.json"),
        output_dir=Path("/tmp/unit_test_top1"),
    )
    assert cfg.primary_metric == "recall@1"


def test_preset_sharpens_the_loss() -> None:
    """Change 1: the loss must not fall back to library defaults."""
    from twotower_top1_optimised.config import build_config

    cfg = build_config(
        "voyage-4-nano",
        run_id="unit_test2",
        rows_path=Path("/tmp/rows.json"),
        output_dir=Path("/tmp/unit_test_top1b"),
    )
    assert cfg.extra["loss_scale"] > 20.0, "scale must exceed the library default of 20.0"
    assert cfg.extra["hardness_mode"] == "hard_negatives"
    assert cfg.extra["hardness_strength"] > 0.0


def test_arm_a_hyperparameters_are_held_fixed() -> None:
    """Everything except the two changes must match Arm A, or attribution fails."""
    from twotower_top1_optimised.config import build_config

    cfg = build_config(
        "voyage-4-nano",
        run_id="unit_test3",
        rows_path=Path("/tmp/rows.json"),
        output_dir=Path("/tmp/unit_test_top1c"),
    )
    assert cfg.train_batch_size == 6
    assert cfg.gradient_accumulation_steps == 2
    assert cfg.train_batch_size * cfg.gradient_accumulation_steps == 12
    assert cfg.learning_rate == pytest.approx(2e-4)
    assert cfg.epochs == 5
    assert cfg.save_total_limit == 5
    assert cfg.truncate_dim == 1024  # nano's native width — a no-op, not truncation


def test_loss_accepts_the_hardness_arguments() -> None:
    """The installed sentence-transformers must actually support the knobs."""
    import inspect

    try:
        from sentence_transformers.sentence_transformer.losses import (
            MultipleNegativesRankingLoss,
        )
    except ImportError:
        from sentence_transformers.losses import MultipleNegativesRankingLoss

    params = inspect.signature(MultipleNegativesRankingLoss.__init__).parameters
    for name in ("scale", "hardness_mode", "hardness_strength"):
        assert name in params, (
            f"installed sentence-transformers has no {name!r} on "
            f"MultipleNegativesRankingLoss; this arm cannot run as designed."
        )


def test_dev_corpus_dedupes_and_prefers_positive_text() -> None:
    """The dev corpus must be unique by candidate id, positives inserted first."""
    from twotower_top1_optimised.data import MultiNegRow
    from twotower_top1_optimised.eval_dev import build_dev_corpus

    rows = [
        MultiNegRow(
            query_key="q1", seeker_id="s1", positive_id="c1", negative_ids=["c2"],
            anchor="a1", positive="POS_C1", negatives=["NEG_C2"],
            n_unique_negatives=1, padded_count=0,
        ),
        MultiNegRow(
            query_key="q2", seeker_id="s2", positive_id="c2", negative_ids=["c1", "c3"],
            anchor="a2", positive="POS_C2", negatives=["NEG_C1", "NEG_C3"],
            n_unique_negatives=2, padded_count=0,
        ),
    ]
    ids, texts = build_dev_corpus(rows)
    assert len(ids) == len(set(ids)) == 3
    mapping = dict(zip(ids, texts))
    # c1 and c2 each appear as both a positive and a negative; the positive-side
    # text must win, matching twotower.eval.build_candidate_corpus.
    assert mapping["c1"] == "POS_C1"
    assert mapping["c2"] == "POS_C2"
    assert mapping["c3"] == "NEG_C3"
