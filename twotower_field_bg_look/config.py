"""Config: `top1_ctrl`'s exact recipe, no new knob at all.

`MODEL_PRESETS` reproduces `top1_ctrl_001`'s settings field-for-field
(`twotower_top1_optimised/config.py`'s preset with `loss_scale=20.0`,
`hardness_mode=None` — the control corner, not the sharpened `top1_001`
arm): rank 8 / alpha 16 / dropout 0.05 on q/k/v/o_proj, micro-batch 6,
accum 2 (effective batch 12), lr 2e-4, `primary_metric="recall@1"`. Pinned
by `tests/test_field_bg_look.py::test_preset_matches_top1_ctrl`.

Unlike `twotower_top1_optimised` (which exposes hardness_mode/
hardness_strength for its own sharpened arm) or `twotower_kl_reg` (which
adds `kl_weight`), this package has no new loss knob — the only thing that
varies here is the row file's text, decided upstream by
`scripts/build_rrf_multineg_triplets_bg_look.py`, not by anything in
`TrainConfig`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from twotower.config import TrainConfig

MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "voyage-4-nano": {
        "model_name": "voyageai/voyage-4-nano",
        "trust_remote_code": True,
        "truncate_dim": 1024,
        "max_seq_length": 4096,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "lora_target_modules": ("q_proj", "k_proj", "v_proj", "o_proj"),
        "expected_layers_per_target": 12,
        # top1_ctrl's winning corner, held fixed.
        "train_batch_size": 6,
        "eval_batch_size": 6,
        "gradient_accumulation_steps": 2,
        "learning_rate": 2e-4,
        "save_total_limit": 5,
        "primary_metric": "recall@1",
        "query_prompt": "Represent the query for retrieving supporting documents: ",
        "document_prompt": "Represent the document for retrieval: ",
    },
}

EFFECTIVE_BATCH_TARGET = 12
DEFAULT_LOSS_SCALE = 20.0  # library default, pinned to top1_ctrl's value


def build_config(
    preset: str,
    *,
    run_id: str,
    rows_path: Path,
    negatives_per_anchor: int = 1,
    epochs: int = 5,
    seed: int = 42,
    loss_scale: float = DEFAULT_LOSS_SCALE,
    **overrides: Any,
) -> TrainConfig:
    """Build a TrainConfig for this arm; never touches a prior experiment's dir."""
    if preset not in MODEL_PRESETS:
        raise KeyError(f"unknown preset {preset!r}; choices: {sorted(MODEL_PRESETS)}")

    fields = dict(MODEL_PRESETS[preset])
    fields.update(overrides)
    fields.setdefault("epochs", epochs)
    fields.setdefault("seed", seed)
    fields.setdefault("output_dir", Path("artifacts/twotower_field_bg_look") / run_id)
    extra = dict(fields.get("extra") or {})
    extra.update(
        {
            "rows_path": str(rows_path),
            "negatives_per_anchor": negatives_per_anchor,
            "loss_scale": loss_scale,
        }
    )
    fields["extra"] = extra

    cfg = TrainConfig(**fields)
    eff = cfg.train_batch_size * cfg.gradient_accumulation_steps
    if eff != EFFECTIVE_BATCH_TARGET:
        print(
            f"\n{'!' * 78}\nWARNING: effective batch is "
            f"{cfg.train_batch_size}x{cfg.gradient_accumulation_steps}={eff}, not "
            f"{EFFECTIVE_BATCH_TARGET}. This arm will not be comparable to top1_ctrl "
            f"(different optimizer-step count).\n{'!' * 78}\n"
        )
    if cfg.primary_metric != "recall@1":
        print(
            f"\n{'!' * 78}\nWARNING: primary_metric is {cfg.primary_metric!r}, not "
            f"'recall@1' — checkpoint selection will not match top1_ctrl's rule."
            f"\n{'!' * 78}\n"
        )
    return cfg
