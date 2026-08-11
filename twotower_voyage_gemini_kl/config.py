"""Config: `voyage_gemini_ctrl_001`'s exact recipe plus `twotower_kl_reg`'s
KL-leash knob, nothing else varied.

`MODEL_PRESETS` below reproduces `twotower_voyage_gemini_ctrl/config.py`'s
preset field-for-field (`loss_scale=20.0`, `hardness_mode=None` — the control
corner, same as `top1_ctrl`): rank 8 / alpha 16 / dropout 0.05 on q/k/v/o_proj,
micro-batch 6, accum 2 (effective batch 12), lr 2e-4,
`primary_metric="recall@1"`. Pinned by
`tests/test_voyage_gemini_kl.py::test_preset_matches_voyage_gemini_ctrl`.

The only new knob is `kl_weight`, copied unchanged from `twotower_kl_reg`
(`DEFAULT_KL_WEIGHT = 0.5`) — this experiment retests that exact mechanism on
a bigger, different batch, not a differently-tuned one.
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
# `extra` and are recorded in run_meta.json for every run. scale is
# voyage_gemini_ctrl's/top1_ctrl's exact value (library default); kl_weight is
# twotower_kl_reg's exact value, unchanged, so this is a true retest of the
# same mechanism on new data.
DEFAULT_LOSS_SCALE = 20.0
DEFAULT_KL_WEIGHT = 0.5


def build_config(
    preset: str,
    *,
    run_id: str,
    rows_path: Path,
    negatives_per_anchor: int = 1,
    epochs: int = 5,
    seed: int = 42,
    loss_scale: float = DEFAULT_LOSS_SCALE,
    kl_weight: float = DEFAULT_KL_WEIGHT,
    **overrides: Any,
) -> TrainConfig:
    """Build a TrainConfig for this arm; never touches a prior experiment's dir."""
    if preset not in MODEL_PRESETS:
        raise KeyError(f"unknown preset {preset!r}; choices: {sorted(MODEL_PRESETS)}")

    fields = dict(MODEL_PRESETS[preset])
    fields.update(overrides)
    fields.setdefault("epochs", epochs)
    fields.setdefault("seed", seed)
    fields.setdefault("output_dir", Path("artifacts/twotower_voyage_gemini_kl") / run_id)
    extra = dict(fields.get("extra") or {})
    extra.update(
        {
            "rows_path": str(rows_path),
            "negatives_per_anchor": negatives_per_anchor,
            "loss_scale": loss_scale,
            "kl_weight": kl_weight,
        }
    )
    fields["extra"] = extra

    cfg = TrainConfig(**fields)
    eff = cfg.train_batch_size * cfg.gradient_accumulation_steps
    if eff != EFFECTIVE_BATCH_TARGET:
        print(
            f"\n{'!' * 78}\nWARNING: effective batch is "
            f"{cfg.train_batch_size}x{cfg.gradient_accumulation_steps}={eff}, not "
            f"{EFFECTIVE_BATCH_TARGET}. Not comparable to voyage_gemini_ctrl_001 "
            f"(different optimizer-step count).\n{'!' * 78}\n"
        )
    if cfg.primary_metric != "recall@1":
        print(
            f"\n{'!' * 78}\nWARNING: primary_metric is {cfg.primary_metric!r}, not "
            f"'recall@1' — checkpoint selection will not match "
            f"voyage_gemini_ctrl_001's rule.\n{'!' * 78}\n"
        )
    return cfg
