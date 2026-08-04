"""Pin twotower_kl_reg's copies and its one load-bearing change.

The isolation rule requires copying rather than importing a prior
experiment's code and pinning each copy so the two cannot drift. `data.py`
and `eval_dev.py` are copies of `twotower_top1_optimised`'s — if they drift,
this arm stops training/selecting on exactly what `top1_ctrl` did, breaking
the comparison this experiment exists to make.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_data_py_matches_top1_optimised() -> None:
    copy = REPO / "twotower_kl_reg" / "data.py"
    orig = REPO / "twotower_top1_optimised" / "data.py"
    normalised = orig.read_text(encoding="utf-8")
    copy_text = copy.read_text(encoding="utf-8")
    # Both files carry only cosmetic docstring differences explaining why each
    # copy exists — the actual code (everything after the module docstring)
    # must be byte-identical.
    copy_code = copy_text.split('"""', 2)[2]
    orig_code = normalised.split('"""', 2)[2]
    assert copy_code == orig_code, (
        "data.py's code has drifted from twotower_top1_optimised's — this arm "
        "would no longer train on the same rows as top1_ctrl, breaking the "
        "comparison."
    )


def test_eval_dev_py_matches_top1_optimised() -> None:
    copy = (REPO / "twotower_kl_reg" / "eval_dev.py").read_text(encoding="utf-8")
    orig = (REPO / "twotower_top1_optimised" / "eval_dev.py").read_text(encoding="utf-8")
    normalised = copy.replace("twotower_kl_reg", "twotower_top1_optimised")
    copy_code = normalised.split('"""', 2)[2]
    orig_code = orig.split('"""', 2)[2]
    assert copy_code == orig_code, (
        "eval_dev.py's code has drifted from twotower_top1_optimised's — "
        "checkpoint selection would no longer match top1_ctrl's rule."
    )


def test_preset_matches_top1_ctrl() -> None:
    """Every knob except the new loss term must equal top1_ctrl's exact config."""
    from twotower_kl_reg.config import build_config

    cfg = build_config(
        "voyage-4-nano",
        run_id="unit_test",
        rows_path=Path("/tmp/rows.json"),
        output_dir=Path("/tmp/unit_test_kl_reg"),
    )
    assert cfg.lora_rank == 8
    assert cfg.lora_alpha == 16
    assert cfg.lora_dropout == 0.05
    assert cfg.lora_target_modules == ("q_proj", "k_proj", "v_proj", "o_proj")
    assert cfg.train_batch_size == 6
    assert cfg.gradient_accumulation_steps == 2
    assert cfg.train_batch_size * cfg.gradient_accumulation_steps == 12
    assert cfg.learning_rate == 2e-4
    assert cfg.epochs == 5
    assert cfg.primary_metric == "recall@1"
    assert cfg.extra["loss_scale"] == 20.0, "must match top1_ctrl's library-default scale, not top1_sharp's 50.0"


def test_kl_weight_is_configurable_and_defaults_positive() -> None:
    from twotower_kl_reg.config import build_config

    cfg = build_config(
        "voyage-4-nano",
        run_id="unit_test2",
        rows_path=Path("/tmp/rows.json"),
        output_dir=Path("/tmp/unit_test_kl_reg2"),
    )
    assert cfg.extra["kl_weight"] > 0.0

    cfg2 = build_config(
        "voyage-4-nano",
        run_id="unit_test3",
        rows_path=Path("/tmp/rows.json"),
        output_dir=Path("/tmp/unit_test_kl_reg3"),
        kl_weight=1.25,
    )
    assert cfg2.extra["kl_weight"] == 1.25
