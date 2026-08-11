"""Pin twotower_qwen_voyage_gemini's deliberate duplicates and config against
twotower_qwen_bigbatch's winning micro-6 preset, and the row file against its
known population.

The isolation rule requires copying rather than editing a prior experiment's
code, and requires pinning each copy with a test so the two cannot silently
drift. ``data.py``, ``eval_dev.py``, ``model.py``, and ``checkpoint.py`` were
copied from ``twotower_qwen_bigbatch/`` with nothing changed but the package
name; if someone edits one and not the other, this run stops being comparable
to ``qwen_micro6_r1``. Same discipline as
``tests/test_qwen_bigbatch_copies.py`` one package upstream.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from twotower_qwen_voyage_gemini.train import _find_resumable_checkpoint, resolve_resume_checkpoint

REPO = Path(__file__).resolve().parent.parent

# (copy, original) — both live directly under twotower_qwen_bigbatch/, unlike
# the upstream package's own copies (which trace further back to
# twotower_rrf_triplet_ablation/ and twotower_rrf_triplet/).
BYTE_IDENTICAL_COPIES = [
    "data.py",
    "eval_dev.py",
    "model.py",
    "checkpoint.py",
]


def _normalise(text: str) -> str:
    """Rename the copy's package back to the original's, so only real edits differ."""
    return text.replace("twotower_qwen_voyage_gemini", "twotower_qwen_bigbatch")


@pytest.mark.parametrize("filename", BYTE_IDENTICAL_COPIES)
def test_copy_matches_qwen_bigbatch(filename: str) -> None:
    copy_path = REPO / "twotower_qwen_voyage_gemini" / filename
    orig_path = REPO / "twotower_qwen_bigbatch" / filename
    assert copy_path.exists(), f"missing copy {copy_path}"
    assert orig_path.exists(), f"missing original {orig_path}"

    copy_text = _normalise(copy_path.read_text(encoding="utf-8"))
    orig_text = orig_path.read_text(encoding="utf-8")
    assert copy_text == orig_text, (
        f"{filename} has drifted from twotower_qwen_bigbatch/{filename}. If the "
        f"change is intentional for this experiment, move it out of the copied "
        f"file (or update this test deliberately) — silent drift breaks "
        f"comparability with twotower_qwen_bigbatch's qwen_micro6_r1."
    )


def test_effective_batch_target_matches_qwen_bigbatch() -> None:
    """Both experiments must pin the same effective batch, or the runs aren't
    comparable (different optimizer-step counts)."""
    from twotower_qwen_bigbatch.config import EFFECTIVE_BATCH_TARGET as upstream_target
    from twotower_qwen_voyage_gemini.config import EFFECTIVE_BATCH_TARGET as this_target

    assert this_target == upstream_target == 12


def test_prompts_match_qwen_bigbatch() -> None:
    """Text packing must be byte-identical to every prior Qwen run's."""
    from twotower_qwen_bigbatch.config import QWEN3_DOCUMENT_PROMPT as upstream_doc
    from twotower_qwen_bigbatch.config import QWEN3_QUERY_PROMPT as upstream_query
    from twotower_qwen_voyage_gemini.config import QWEN3_DOCUMENT_PROMPT as this_doc
    from twotower_qwen_voyage_gemini.config import QWEN3_QUERY_PROMPT as this_query

    assert this_query == upstream_query
    assert this_doc == upstream_doc


def test_preset_matches_qwen_bigbatch_micro6() -> None:
    """Every LoRA/optimizer knob must equal qwen_micro6_r1's exact config —
    the only variable in this experiment is the training rows."""
    from twotower_qwen_bigbatch.config import build_config as upstream_build_config
    from twotower_qwen_voyage_gemini.config import build_config

    upstream_cfg = upstream_build_config(
        "qwen3-8b",
        run_id="unit_test_upstream",
        rows_path=Path("/tmp/rows.json"),
        output_dir=Path("/tmp/unit_test_upstream_unused"),
        train_batch_size=6,
        eval_batch_size=6,
        gradient_accumulation_steps=2,
    )
    cfg = build_config(
        "qwen3-8b",
        run_id="unit_test",
        rows_path=Path("/tmp/rows.json"),
        output_dir=Path("/tmp/unit_test_unused"),
    )

    # this package's preset already defaults to micro-6 (no override needed)
    assert cfg.train_batch_size == upstream_cfg.train_batch_size == 6
    assert cfg.eval_batch_size == upstream_cfg.eval_batch_size == 6
    assert cfg.gradient_accumulation_steps == upstream_cfg.gradient_accumulation_steps == 2
    assert cfg.train_batch_size * cfg.gradient_accumulation_steps == 12

    for field in (
        "model_name",
        "trust_remote_code",
        "truncate_dim",
        "max_seq_length",
        "lora_rank",
        "lora_alpha",
        "lora_target_modules",
        "expected_layers_per_target",
        "learning_rate",
        "epochs",
        "save_total_limit",
        "query_prompt",
        "document_prompt",
    ):
        assert getattr(cfg, field) == getattr(upstream_cfg, field), field

    assert cfg.model_name == "Qwen/Qwen3-Embedding-8B"
    assert cfg.lora_rank == 8
    assert cfg.lora_alpha == 16
    assert cfg.lora_target_modules == ("q_proj", "k_proj", "v_proj", "o_proj")
    assert cfg.expected_layers_per_target == 36
    assert cfg.truncate_dim == 1024
    assert cfg.max_seq_length == 4096
    assert cfg.learning_rate == 1e-4
    assert cfg.epochs == 5
    assert cfg.save_total_limit == 5
    assert cfg.extra["torch_dtype"] == "bfloat16"
    assert cfg.extra["gradient_checkpointing_override"] is True


def test_overriding_extra_cannot_drop_memory_settings() -> None:
    """A caller passing its own `extra` must not silently disable bf16."""
    from twotower_qwen_voyage_gemini.config import build_config

    cfg = build_config(
        "qwen3-8b",
        run_id="unit_test2",
        rows_path=Path("/tmp/rows.json"),
        output_dir=Path("/tmp/unit_test_unused2"),
        extra={"something_else": 1},
    )
    assert cfg.extra["torch_dtype"] == "bfloat16"
    assert cfg.extra["gradient_checkpointing_override"] is True
    assert cfg.extra["something_else"] == 1


def test_row_file_population() -> None:
    """3,008 rows / 1,921 seekers / 0% padding, from smoke_test_002's 2,187
    both-class query_keys (k=1) — the same row file
    twotower_voyage_gemini_ctrl trained voyage-4-nano on, mounted read-only
    rather than copied or regenerated."""
    path = (
        REPO
        / "artifacts"
        / "twotower_voyage_gemini_ctrl"
        / "voyage_gemini_smoke002_multineg_k1.json"
    )
    if not path.exists():
        pytest.skip(
            "row file not built yet — see twotower_voyage_gemini_ctrl's own "
            "build command"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload["summary"]
    assert summary["n_rows"] == 3008
    assert summary["n_seekers"] == 1921
    assert summary["total_padded_negative_slots"] == 0
    assert summary["query_keys_with_both_classes"] == 2187
    assert summary["negatives_per_anchor"] == 1
    assert len(payload["rows"]) == 3008
    seekers = {r["seeker_id"] for r in payload["rows"]}
    assert len(seekers) == 1921


def test_guard_recognizes_this_rows_source() -> None:
    """eval_real_full.guard must accept this package's rows_path token (it
    already does — twotower_voyage_gemini_ctrl registered it first, and this
    package reuses the identical filename), or scoring on all 200 real pairs
    will fail loudly at eval time. Read-only check; this package does not
    modify guard.py."""
    from eval_real_full.guard import SYNTHETIC_ONLY_ROW_SOURCES

    rows_path = "/root/rows/voyage_gemini_smoke002_multineg_k1.json"
    assert any(token in rows_path for token in SYNTHETIC_ONLY_ROW_SOURCES)


# ---------------------------------------------------------------------------
# Preemption-resume logic (B200 fix). These pin the actual failure sequence
# hit during launch: two B200 attempts were killed by Modal's preemption
# auto-restart ("on the same input", see
# https://modal.com/docs/guide/preemption); the second attempt had already
# completed 13 real training steps before being killed. run_training() used to
# treat any non-empty output_dir as a hard error, which meant the auto-restart
# always crashed against its own prior partial state instead of resuming.
# resolve_resume_checkpoint()/_find_resumable_checkpoint() are exercised here
# directly (no model loading, no GPU) so this fix is verified before spending
# more GPU time on it.
# ---------------------------------------------------------------------------


def test_find_resumable_checkpoint_missing_dir(tmp_path: Path) -> None:
    assert _find_resumable_checkpoint(tmp_path / "does_not_exist") is None


def test_find_resumable_checkpoint_empty_checkpoints_dir(tmp_path: Path) -> None:
    (tmp_path / "checkpoints").mkdir()
    assert _find_resumable_checkpoint(tmp_path) is None


def test_find_resumable_checkpoint_ignores_incomplete_checkpoint(tmp_path: Path) -> None:
    """A checkpoint-N dir without trainer_state.json means the save never
    finished (e.g. killed mid-write) — must not be treated as resumable."""
    ckpt = tmp_path / "checkpoints" / "checkpoint-50"
    ckpt.mkdir(parents=True)
    (ckpt / "some_partial_file.bin").write_text("x")
    assert _find_resumable_checkpoint(tmp_path) is None


def test_find_resumable_checkpoint_picks_latest_step(tmp_path: Path) -> None:
    for step in (50, 226, 113):
        ckpt = tmp_path / "checkpoints" / f"checkpoint-{step}"
        ckpt.mkdir(parents=True)
        (ckpt / "trainer_state.json").write_text("{}")
    result = _find_resumable_checkpoint(tmp_path)
    assert result == str(tmp_path / "checkpoints" / "checkpoint-226")


def test_resolve_resume_checkpoint_fresh_dir_is_fresh_start(tmp_path: Path) -> None:
    """A run-id never used before: output_dir doesn't exist yet."""
    assert resolve_resume_checkpoint(tmp_path / "brand_new_run", None) is None


def test_resolve_resume_checkpoint_empty_existing_dir_is_fresh_start(tmp_path: Path) -> None:
    assert resolve_resume_checkpoint(tmp_path, None) is None


def test_resolve_resume_checkpoint_raises_on_junk_only(tmp_path: Path) -> None:
    """Preempted before the first checkpoint save (e.g. killed at step 13 of
    ~226, as actually happened) — only run_meta.json exists, no
    checkpoints/checkpoint-*/trainer_state.json yet. Must still fail loudly:
    there is genuinely nothing to resume from, and silently starting over
    would be surprising (better to require a fresh run-id explicitly)."""
    (tmp_path / "run_meta.json").write_text("{}")
    with pytest.raises(FileExistsError, match="no valid HF checkpoint"):
        resolve_resume_checkpoint(tmp_path, None)


def test_resolve_resume_checkpoint_auto_detects_valid_checkpoint(tmp_path: Path) -> None:
    """The actual fix: preempted after epoch 1's checkpoint save landed —
    must resume from it instead of raising."""
    ckpt = tmp_path / "checkpoints" / "checkpoint-226"
    ckpt.mkdir(parents=True)
    (ckpt / "trainer_state.json").write_text("{}")
    (tmp_path / "run_meta.json").write_text("{}")

    result = resolve_resume_checkpoint(tmp_path, None)
    assert result == str(ckpt)


def test_resolve_resume_checkpoint_explicit_path_wins_over_detection(tmp_path: Path) -> None:
    """An explicit --resume-from-checkpoint must be honored as-is, even if
    auto-detection would have picked a different (e.g. later) checkpoint."""
    for step in (100, 226):
        ckpt = tmp_path / "checkpoints" / f"checkpoint-{step}"
        ckpt.mkdir(parents=True)
        (ckpt / "trainer_state.json").write_text("{}")

    explicit = str(tmp_path / "checkpoints" / "checkpoint-100")
    result = resolve_resume_checkpoint(tmp_path, explicit)
    assert result == explicit
