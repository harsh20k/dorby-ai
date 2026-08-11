"""Pin twotower_queryonly_back_look's copies and its config.

Seeker = search query only (no profile fields), candidate =
background+lookingFor — the recall@1-best combo found by the top1_ctrl
105-way field/query sweep, trained here for the first time.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_data_py_matches_top1_optimised() -> None:
    copy = REPO / "twotower_queryonly_back_look" / "data.py"
    orig = REPO / "twotower_top1_optimised" / "data.py"
    copy_text = copy.read_text(encoding="utf-8")
    orig_text = orig.read_text(encoding="utf-8")
    copy_code = copy_text.split('"""', 2)[2]
    orig_code = orig_text.split('"""', 2)[2]
    assert copy_code == orig_code, (
        "data.py's code has drifted from twotower_top1_optimised's — this arm "
        "would no longer carve train/dev the same way top1_ctrl did."
    )


def test_eval_dev_py_matches_top1_optimised() -> None:
    copy = (REPO / "twotower_queryonly_back_look" / "eval_dev.py").read_text(encoding="utf-8")
    orig = (REPO / "twotower_top1_optimised" / "eval_dev.py").read_text(encoding="utf-8")
    normalised = copy.replace("twotower_queryonly_back_look", "twotower_top1_optimised")
    copy_code = normalised.split('"""', 2)[2]
    orig_code = orig.split('"""', 2)[2]
    assert copy_code == orig_code, (
        "eval_dev.py's code has drifted from twotower_top1_optimised's — "
        "checkpoint selection would no longer match top1_ctrl's rule."
    )


def test_preset_matches_top1_ctrl() -> None:
    """Every knob must equal top1_ctrl's exact config — this package adds no new loss knob."""
    from twotower_queryonly_back_look.config import build_config

    cfg = build_config(
        "voyage-4-nano",
        run_id="unit_test",
        rows_path=Path("/tmp/rows.json"),
        output_dir=Path("/tmp/unit_test_queryonly_back_look"),
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
    assert cfg.extra["loss_scale"] == 20.0, "must match top1_ctrl's library-default scale, not top1_sharp's 50.0"


def test_seeker_is_query_only_candidate_is_background_lookingfor() -> None:
    from field_pairs_sweep.text import background_lookingfor
    from query_weighted.text import query_only

    profile = {
        "positioning": "Founder of a climate-tech startup.",
        "background": "Ex-Google, 10 years in ML infra.",
        "lookingFor": "Intros to seed investors.",
        "notes": "Loves rock climbing.",
    }
    seeker_text = query_only(profile, "Looking for seed investors in climate tech")
    assert seeker_text == "Search query: Looking for seed investors in climate tech"
    assert "positioning:" not in seeker_text
    assert "background:" not in seeker_text

    cand_text = background_lookingfor(profile)
    assert "background:" in cand_text
    assert "lookingFor:" in cand_text
    assert "positioning:" not in cand_text


def test_row_file_matches_top1_ctrl_population() -> None:
    """Same 643 rows / 297 seekers / 0% padding as the row file top1_ctrl trained on."""
    import json

    path = REPO / "artifacts" / "twotower_queryonly_back_look" / "rrf_003_multineg_k1_queryonly_back_look.json"
    if not path.exists():
        import pytest

        pytest.skip("row file not built yet — run scripts/build_rrf_multineg_triplets_queryonly_back_look.py")
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload["summary"]
    assert summary["n_rows"] == 643
    assert summary["n_seekers"] == 297
    assert summary["total_padded_negative_slots"] == 0
    assert summary["seeker_fields"] == []
    assert summary["seeker_uses_query"] is True
    assert summary["candidate_fields"] == ["background", "lookingFor"]
