"""Config for the query-only arm: exact top1_ctrl recipe, seeker side is the
search query alone — no profile text at all.

Third leg of a three-way seeker-text comparison: `twotower_top1_optimised/
top1_ctrl` trains on profile+query concatenated; `twotower_no_query/` trains
on profile only; this package trains on query only. Motivation:
`twotower_query_weighted/` found that swapping `top1_ctrl`'s seeker text to
query-only *at eval time only* (no retraining) roughly doubles recall@1
(0.19 -> 0.32), and `twotower_no_query/` then found that training condition
barely moves profile-only performance (`no_query_001` landed within noise of
`top1_ctrl`'s own eval-time profile-only swap). This package asks the same
question for the query side: does actually training on query-only text beat
just swapping to query-only text on an already-trained model?

Every hyperparameter here is `top1_ctrl`'s actual launch config (recorded in
`artifacts/twotower_top1_optimised/top1_ctrl_001/run_meta.json`): library-default
`MultipleNegativesRankingLoss` (`scale=20.0`, no hardness weighting),
`primary_metric="recall@1"` via `CorpusRecallDevEvaluator`, micro-batch 6 /
accum 2 (effective batch 12, 245 optimizer steps), lr 2e-4, 5 epochs. The only
difference by construction: rows come from
`rrf_003_multineg_k1_query_only.json`
(`scripts/build_rrf_multineg_triplets_query_only.py`), whose anchor text is
`query_weighted.text.query_only` instead of `seeker_to_text` — verified
row-for-row identical to the original k1 file on every field except `anchor`
(643/643 rows differ only there; ids, positives, negatives are
byte-identical; 0 rows needed the empty-query fallback to profile text).
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
    fields.setdefault("output_dir", Path("artifacts/twotower_query_only") / run_id)
    extra = dict(fields.get("extra") or {})
    extra.update(
        {
            "rows_path": str(rows_path),
            "negatives_per_anchor": negatives_per_anchor,
            "loss_scale": loss_scale,
            "hardness_mode": hardness_mode,
            "hardness_strength": hardness_strength,
            "anchor_text": "query_only (no profile)",
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
