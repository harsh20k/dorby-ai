"""Pin twotower_voyage_gemini_ctrl's copies and its config.

top1_ctrl's exact recipe, retrained on the pairing_voyage_gemini batch instead
of rrf_003 — see twotower_voyage_gemini_ctrl/__init__.py for the full
rationale and the leakage caveat found on the new batch before training.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_data_py_matches_top1_optimised() -> None:
    copy = REPO / "twotower_voyage_gemini_ctrl" / "data.py"
    orig = REPO / "twotower_top1_optimised" / "data.py"
    copy_code = copy.read_text(encoding="utf-8").split('"""', 2)[2]
    orig_code = orig.read_text(encoding="utf-8").split('"""', 2)[2]
    assert copy_code == orig_code, (
        "data.py's code has drifted from twotower_top1_optimised's — this arm "
        "would no longer carve train/dev the same way top1_ctrl did."
    )


def test_eval_dev_py_matches_top1_optimised() -> None:
    copy = (REPO / "twotower_voyage_gemini_ctrl" / "eval_dev.py").read_text(encoding="utf-8")
    orig = (REPO / "twotower_top1_optimised" / "eval_dev.py").read_text(encoding="utf-8")
    normalised = copy.replace("twotower_voyage_gemini_ctrl", "twotower_top1_optimised")
    copy_code = normalised.split('"""', 2)[2]
    orig_code = orig.split('"""', 2)[2]
    assert copy_code == orig_code, (
        "eval_dev.py's code has drifted from twotower_top1_optimised's — "
        "checkpoint selection would no longer match top1_ctrl's rule."
    )


def test_preset_matches_top1_ctrl() -> None:
    """Every knob must equal top1_ctrl_001's exact config."""
    from twotower_voyage_gemini_ctrl.config import build_config

    cfg = build_config(
        "voyage-4-nano",
        run_id="unit_test",
        rows_path=Path("/tmp/rows.json"),
        output_dir=Path("/tmp/unit_test_voyage_gemini_ctrl"),
    )
    assert cfg.lora_rank == 8
    assert cfg.lora_alpha == 16
    assert cfg.lora_dropout == 0.05
    assert cfg.lora_target_modules == ("q_proj", "k_proj", "v_proj", "o_proj")
    assert cfg.truncate_dim == 1024
    assert cfg.max_seq_length == 4096
    assert cfg.train_batch_size == 6
    assert cfg.gradient_accumulation_steps == 2
    assert cfg.train_batch_size * cfg.gradient_accumulation_steps == 12
    assert cfg.learning_rate == 2e-4
    assert cfg.epochs == 5
    assert cfg.primary_metric == "recall@1"
    assert cfg.extra["loss_scale"] == 20.0
    assert cfg.extra["hardness_mode"] is None


def test_row_file_population() -> None:
    """3,008 rows / 1,921 seekers / 0% padding, from smoke_test_002's 2,187
    both-class query_keys (k=1)."""
    import json

    path = REPO / "artifacts" / "twotower_voyage_gemini_ctrl" / "voyage_gemini_smoke002_multineg_k1.json"
    if not path.exists():
        import pytest

        pytest.skip(
            "row file not built yet — run scripts/build_rrf_multineg_triplets.py "
            "--batch-dir artifacts/pairing_voyage_gemini/smoke_test_002"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload["summary"]
    assert summary["n_rows"] == 3008
    assert summary["n_seekers"] == 1921
    assert summary["total_padded_negative_slots"] == 0
    assert summary["query_keys_with_both_classes"] == 2187
    assert summary["negatives_per_anchor"] == 1


def test_guard_recognizes_this_rows_source() -> None:
    """eval_real_full.guard must accept this package's rows_path token, or
    scoring on all 200 real pairs will fail loudly at eval time."""
    from eval_real_full.guard import SYNTHETIC_ONLY_ROW_SOURCES

    rows_path = "/root/rows/voyage_gemini_smoke002_multineg_k1.json"
    assert any(token in rows_path for token in SYNTHETIC_ONLY_ROW_SOURCES)
