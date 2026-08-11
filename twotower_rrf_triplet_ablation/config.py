"""Model preset + config factory for the batch-vs-negatives ablation.

Reuses twotower.config.TrainConfig as-is (unmodified, frozen dataclass), same
convention as the two prior triplet packages. Only adds this experiment's
preset defaults and its own output-path convention
(artifacts/twotower_rrf_triplet_ablation/<run_id>).

Preset defaults encode the *baseline* corner (micro 2 / accum 6 / k=1); every
arm overrides what it varies via CLI. Effective batch is
train_batch_size * gradient_accumulation_steps and must stay 12 in every arm —
see the package docstring for why.
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
        "lora_target_modules": ("q_proj", "k_proj", "v_proj", "o_proj"),
        "expected_layers_per_target": 12,
        # baseline corner; arms override micro-batch/accum to keep eff batch 12
        "train_batch_size": 2,
        "eval_batch_size": 6,
        "gradient_accumulation_steps": 6,
        "learning_rate": 2e-4,
        # 5 epochs + eval-per-epoch means the default limit of 3 prunes the
        # epoch-1/2 checkpoints before select_best_checkpoint runs. Both prior
        # experiments silently shipped a final-epoch model for exactly this
        # reason; keeping all 5 lets selection actually pick what it chose.
        "save_total_limit": 5,
        "query_prompt": "Represent the query for retrieving supporting documents: ",
        "document_prompt": "Represent the document for retrieval: ",
    },
}

EFFECTIVE_BATCH_TARGET = 12


def build_config(
    preset: str,
    *,
    run_id: str,
    rows_path: Path,
    negatives_per_anchor: int = 1,
    epochs: int = 5,
    seed: int = 42,
    **overrides: Any,
) -> TrainConfig:
    """Build a TrainConfig for this ablation; never touches the prior
    experiments' output dirs."""
    if preset not in MODEL_PRESETS:
        raise KeyError(f"unknown preset {preset!r}; choices: {sorted(MODEL_PRESETS)}")

    fields = dict(MODEL_PRESETS[preset])
    fields.update(overrides)
    fields.setdefault("epochs", epochs)
    fields.setdefault("seed", seed)
    fields.setdefault("output_dir", Path("artifacts/twotower_rrf_triplet_ablation") / run_id)
    fields.setdefault("extra", {}).update(
        {"rows_path": str(rows_path), "negatives_per_anchor": negatives_per_anchor}
    )

    cfg = TrainConfig(**fields)
    eff = cfg.train_batch_size * cfg.gradient_accumulation_steps
    if eff != EFFECTIVE_BATCH_TARGET:
        # Loud, not fatal: the whole ablation rests on effective batch being
        # matched across arms, so a mismatch must be impossible to miss.
        print(
            f"\n{'!' * 78}\nWARNING: effective batch is "
            f"{cfg.train_batch_size}x{cfg.gradient_accumulation_steps}={eff}, "
            f"not the ablation's target {EFFECTIVE_BATCH_TARGET}. Arms will not "
            f"be comparable (optimizer-step counts will differ).\n{'!' * 78}\n"
        )
    return cfg
