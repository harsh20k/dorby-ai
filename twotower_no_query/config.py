"""Config for the no-query arm: exact top1_ctrl recipe, minus the search query.

`twotower_top1_optimised/top1_ctrl_001` is the best fine-tune in the project so
far (all-200 recall@1 0.19, MRR 0.3550) — but its training rows, like every
other two-tower experiment in this repo, concatenated the search query into
the seeker text (`seeker_to_text`). Separately, `twotower_query_weighted/`
found that swapping to a query-only or query-weighted seeker representation
*at eval time only* (no retraining) roughly doubles that model's recall@1.
This package asks a different question: what if the model is *trained* on
profile-only seeker text from the start, never seeing the query at all?

Every hyperparameter here is `top1_ctrl`'s actual launch config (recorded in
`artifacts/twotower_top1_optimised/top1_ctrl_001/run_meta.json`): library-default
`MultipleNegativesRankingLoss` (`scale=20.0`, no hardness weighting — `top1_ctrl`
was the *control* arm of that package, isolating the checkpoint-selection fix
from the loss-sharpening change), `primary_metric="recall@1"` via
`CorpusRecallDevEvaluator`, micro-batch 6 / accum 2 (effective batch 12, 245
optimizer steps), lr 2e-4, 5 epochs. The only difference by construction: rows
come from `rrf_003_multineg_k1_no_query.json`
(`scripts/build_rrf_multineg_triplets_no_query.py`), whose anchor text is
`profile_to_text` instead of `seeker_to_text` — verified row-for-row identical
to the original k1 file on every field except `anchor` (643/643 rows differ
only there; ids, positives, negatives are byte-identical).
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

# top1_ctrl's actual values (library defaults, not the sharpened top1_001 arm).
DEFAULT_LOSS_SCALE = 20.0
DEFAULT_HARDNESS_MODE = None
DEFAULT_HARDNESS_STRENGTH = 0.0


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
    fields.setdefault("output_dir", Path("artifacts/twotower_no_query") / run_id)
    extra = dict(fields.get("extra") or {})
    extra.update(
        {
            "rows_path": str(rows_path),
            "negatives_per_anchor": negatives_per_anchor,
            "loss_scale": loss_scale,
            "hardness_mode": hardness_mode,
            "hardness_strength": hardness_strength,
            "anchor_text": "profile_only (no searchQuery)",
        }
    )
    fields["extra"] = extra

    cfg = TrainConfig(**fields)
    eff = cfg.train_batch_size * cfg.gradient_accumulation_steps
    if eff != EFFECTIVE_BATCH_TARGET:
        print(
            f"\n{'!' * 78}\nWARNING: effective batch is "
            f"{cfg.train_batch_size}x{cfg.gradient_accumulation_steps}={eff}, not "
            f"{EFFECTIVE_BATCH_TARGET}. Not comparable to top1_ctrl (different "
            f"optimizer-step count).\n{'!' * 78}\n"
        )
    if cfg.primary_metric != "recall@1":
        print(
            f"\n{'!' * 78}\nWARNING: primary_metric is {cfg.primary_metric!r}, not "
            f"'recall@1' — checkpoint selection will not match top1_ctrl's.\n{'!' * 78}\n"
        )
    return cfg
