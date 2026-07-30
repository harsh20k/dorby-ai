"""Pin twotower_qwen_bigbatch's deliberate duplicates against their originals.

The isolation rule requires copying rather than editing a prior experiment's
code, and requires pinning each copy with a test so the two cannot silently
drift. ``data.py`` and ``eval_dev.py`` were copied from
``twotower_rrf_triplet_ablation/`` with nothing changed but the package name; if
someone edits one and not the other, the Qwen arms stop being comparable to
Arm A and nothing else would catch it.

``model.py`` and ``checkpoint.py`` are copies of ``twotower_rrf_triplet``'s bf16
loading path and are pinned the same way.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# (copy, original, package name in the original)
FROM_ABLATION = [
    ("data.py", "twotower_rrf_triplet_ablation"),
    ("eval_dev.py", "twotower_rrf_triplet_ablation"),
]
FROM_TRIPLET = [
    ("model.py", "twotower_rrf_triplet"),
    ("checkpoint.py", "twotower_rrf_triplet"),
]


def _normalise(text: str, origin_pkg: str) -> str:
    """Rename the copy's package back to the original's, so only real edits differ."""
    return text.replace("twotower_qwen_bigbatch", origin_pkg)


@pytest.mark.parametrize("filename,origin_pkg", FROM_ABLATION + FROM_TRIPLET)
def test_copy_matches_original(filename: str, origin_pkg: str) -> None:
    copy_path = REPO / "twotower_qwen_bigbatch" / filename
    orig_path = REPO / origin_pkg / filename
    assert copy_path.exists(), f"missing copy {copy_path}"
    assert orig_path.exists(), f"missing original {orig_path}"

    copy_text = _normalise(copy_path.read_text(encoding="utf-8"), origin_pkg)
    orig_text = orig_path.read_text(encoding="utf-8")
    assert copy_text == orig_text, (
        f"{filename} has drifted from {origin_pkg}/{filename}. If the change is "
        f"intentional for this experiment, move it out of the copied file (or "
        f"update this test deliberately) — silent drift breaks comparability "
        f"with the run this copy was taken from."
    )


def test_effective_batch_target_matches_ablation() -> None:
    """Both experiments must pin the same effective batch, or arms aren't comparable."""
    from twotower_qwen_bigbatch.config import EFFECTIVE_BATCH_TARGET as qwen_target
    from twotower_rrf_triplet_ablation.config import EFFECTIVE_BATCH_TARGET as nano_target

    assert qwen_target == nano_target == 12


def test_qwen_preset_keeps_memory_settings() -> None:
    """bf16 + gradient checkpointing are load-bearing for any micro-batch > 1."""
    from pathlib import Path as _Path

    from twotower_qwen_bigbatch.config import build_config

    cfg = build_config(
        "qwen3-8b",
        run_id="unit_test",
        rows_path=_Path("/tmp/rows.json"),
        output_dir=_Path("/tmp/unit_test_unused"),
    )
    assert cfg.extra["torch_dtype"] == "bfloat16"
    assert cfg.extra["gradient_checkpointing_override"] is True
    assert cfg.expected_layers_per_target == 36
    assert cfg.train_batch_size * cfg.gradient_accumulation_steps == 12


def test_overriding_extra_cannot_drop_memory_settings() -> None:
    """A caller passing its own `extra` must not silently disable bf16."""
    from pathlib import Path as _Path

    from twotower_qwen_bigbatch.config import build_config

    cfg = build_config(
        "qwen3-8b",
        run_id="unit_test2",
        rows_path=_Path("/tmp/rows.json"),
        output_dir=_Path("/tmp/unit_test_unused2"),
        extra={"something_else": 1},
    )
    assert cfg.extra["torch_dtype"] == "bfloat16"
    assert cfg.extra["gradient_checkpointing_override"] is True
    assert cfg.extra["something_else"] == 1
