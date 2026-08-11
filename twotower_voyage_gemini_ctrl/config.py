"""Config: `top1_ctrl`'s exact recipe, no sharp/hardness variant.

Unlike `twotower_top1_optimised`, this package only ever runs the control
corner (plain library-default `MultipleNegativesRankingLoss`), so the
sharp-loss defaults that package carries for its A/B comparison are dropped —
there is no second arm here to compare against.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from twotower.config import TrainConfig

MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "voyage-4-nano": {
        "model_name": "voyageai/voyage-4-nano",
        "trust_remote_code": True,
        "truncate_dim": 1024,  # nano's native width — a no-op, not a truncation
        "max_seq_length": 4096,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "lora_target_modules": ("q_proj", "k_proj", "v_proj", "o_proj"),
        "expected_layers_per_target": 12,
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

# Loss knobs — not part of TrainConfig (shared, unmodified), so they travel in
# `extra` and are recorded in run_meta.json for every run. top1_ctrl's values,
# not the library's raw defaults (which happen to be the same for scale here).
DEFAULT_LOSS_SCALE = 20.0
DEFAULT_HARDNESS_MODE: str | None = None
DEFAULT_HARDNESS_STRENGTH = 1.0  # unused while hardness_mode is None


def build_config(
    preset: str,
    *,
    run_id: str,
    rows_path: Path,
    negatives_per_anchor: int = 1,
    epochs: int = 5,
    seed: int = 42,
    loss_scale: float = DEFAULT_LOSS_SCALE,
    hardness_mode: str | None = DEFAULT_HARDNESS_MODE,
    hardness_strength: float = DEFAULT_HARDNESS_STRENGTH,
    **overrides: Any,
) -> TrainConfig:
    """Build a TrainConfig for this arm; never touches a prior experiment's dir."""
    if preset not in MODEL_PRESETS:
        raise KeyError(f"unknown preset {preset!r}; choices: {sorted(MODEL_PRESETS)}")

    fields = dict(MODEL_PRESETS[preset])
    fields.update(overrides)
    fields.setdefault("epochs", epochs)
    fields.setdefault("seed", seed)
    fields.setdefault("output_dir", Path("artifacts/twotower_voyage_gemini_ctrl") / run_id)
    extra = dict(fields.get("extra") or {})
    extra.update(
        {
            "rows_path": str(rows_path),
            "negatives_per_anchor": negatives_per_anchor,
            "loss_scale": loss_scale,
            "hardness_mode": hardness_mode,
            "hardness_strength": hardness_strength,
        }
    )
    fields["extra"] = extra

    cfg = TrainConfig(**fields)
    eff = cfg.train_batch_size * cfg.gradient_accumulation_steps
    if eff != EFFECTIVE_BATCH_TARGET:
        print(
            f"\n{'!' * 78}\nWARNING: effective batch is "
            f"{cfg.train_batch_size}x{cfg.gradient_accumulation_steps}={eff}, not "
            f"{EFFECTIVE_BATCH_TARGET}. Not comparable to top1_ctrl_001 (different "
            f"optimizer-step count).\n{'!' * 78}\n"
        )
    if cfg.primary_metric != "recall@1":
        print(
            f"\n{'!' * 78}\nWARNING: primary_metric is {cfg.primary_metric!r}, not "
            f"'recall@1' — checkpoint selection will not match top1_ctrl_001's "
            f"rule.\n{'!' * 78}\n"
        )
    return cfg
