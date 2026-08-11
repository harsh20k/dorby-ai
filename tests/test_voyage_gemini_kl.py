"""Pin twotower_voyage_gemini_kl's copies and its config.

`voyage_gemini_ctrl_001`'s exact recipe and rows, plus `twotower_kl_reg`'s
KL-leash-to-frozen-base loss mechanism (`kl_weight=0.5`, unchanged) — see
`twotower_voyage_gemini_kl/__init__.py` for the full rationale. The isolation
rule requires copying rather than importing a prior experiment's code and
pinning each copy so the two cannot drift: `data.py`/`eval_dev.py` are copies
of `twotower_voyage_gemini_ctrl`'s, `losses.py` is a copy of
`twotower_kl_reg`'s. If any of them drift, this arm stops training/selecting
on exactly what its two source experiments did, breaking the comparison this
experiment exists to make.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _code_after_module_docstring(text: str) -> str:
    return text.split('"""', 2)[2]


def test_data_py_matches_voyage_gemini_ctrl() -> None:
    copy = REPO / "twotower_voyage_gemini_kl" / "data.py"
    orig = REPO / "twotower_voyage_gemini_ctrl" / "data.py"
    copy_code = _code_after_module_docstring(copy.read_text(encoding="utf-8"))
    orig_code = _code_after_module_docstring(orig.read_text(encoding="utf-8"))
    assert copy_code == orig_code, (
        "data.py's code has drifted from twotower_voyage_gemini_ctrl's — this "
        "arm would no longer carve train/dev the same way voyage_gemini_ctrl_001 "
        "did."
    )


def test_eval_dev_py_matches_voyage_gemini_ctrl() -> None:
    copy = (REPO / "twotower_voyage_gemini_kl" / "eval_dev.py").read_text(encoding="utf-8")
    orig = (REPO / "twotower_voyage_gemini_ctrl" / "eval_dev.py").read_text(encoding="utf-8")
    normalised = copy.replace("twotower_voyage_gemini_kl", "twotower_voyage_gemini_ctrl")
    copy_code = _code_after_module_docstring(normalised)
    orig_code = _code_after_module_docstring(orig)
    assert copy_code == orig_code, (
        "eval_dev.py's code has drifted from twotower_voyage_gemini_ctrl's — "
        "checkpoint selection would no longer match voyage_gemini_ctrl_001's rule."
    )


def test_losses_py_matches_kl_reg() -> None:
    copy = REPO / "twotower_voyage_gemini_kl" / "losses.py"
    orig = REPO / "twotower_kl_reg" / "losses.py"
    copy_code = _code_after_module_docstring(copy.read_text(encoding="utf-8"))
    orig_code = _code_after_module_docstring(orig.read_text(encoding="utf-8"))
    assert copy_code == orig_code, (
        "losses.py's KL mechanism has drifted from twotower_kl_reg's — this "
        "would no longer be a true retest of the same loss on new data."
    )


def test_preset_matches_voyage_gemini_ctrl() -> None:
    """Every knob except the new KL term must equal voyage_gemini_ctrl_001's exact config."""
    from twotower_voyage_gemini_kl.config import build_config

    cfg = build_config(
        "voyage-4-nano",
        run_id="unit_test",
        rows_path=Path("/tmp/rows.json"),
        output_dir=Path("/tmp/unit_test_voyage_gemini_kl"),
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
    assert cfg.extra["loss_scale"] == 20.0, "must match voyage_gemini_ctrl's library-default scale"


def test_kl_weight_is_configurable_and_defaults_to_kl_reg_value() -> None:
    from twotower_voyage_gemini_kl.config import build_config

    cfg = build_config(
        "voyage-4-nano",
        run_id="unit_test2",
        rows_path=Path("/tmp/rows.json"),
        output_dir=Path("/tmp/unit_test_voyage_gemini_kl2"),
    )
    assert cfg.extra["kl_weight"] == 0.5, "must default to twotower_kl_reg's exact kl_weight for a true retest"

    cfg2 = build_config(
        "voyage-4-nano",
        run_id="unit_test3",
        rows_path=Path("/tmp/rows.json"),
        output_dir=Path("/tmp/unit_test_voyage_gemini_kl3"),
        kl_weight=1.25,
    )
    assert cfg2.extra["kl_weight"] == 1.25


def test_row_file_population() -> None:
    """3,008 rows / 1,921 seekers / 0% padding, from smoke_test_002's 2,187
    both-class query_keys (k=1) — same row file voyage_gemini_ctrl_001 trained
    on, mounted read-only from twotower_voyage_gemini_ctrl's artifacts dir."""
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
