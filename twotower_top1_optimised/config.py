"""Config for the recall@1-targeted arm: sharper loss + recall-aware selection.

Two changes from `twotower_rrf_triplet_ablation`, both aimed squarely at top-1
retrieval and both willing to cost pair AUC:

**1. The loss now concentrates on the top competitor.**
Every prior run instantiated `MultipleNegativesRankingLoss(model=model)` with
pure defaults — `scale=20.0`, `hardness_mode=None`, `hardness_strength=0.0`. The
temperature was never tuned and the library's built-in hard-negative weighting
was never switched on.

`scale` is an inverse temperature on the softmax over candidates. Raising it
makes the distribution peakier, so gradient concentrates on the *highest-scoring*
negative instead of spreading across all of them — which is exactly "beat
whoever is currently ranked first". `hardness_mode="hard_negatives"` with
`hardness_strength > 0` additionally up-weights the explicit hard negative.

**2. Checkpoint selection can finally see recall@1.**
`eval_dev.CorpusRecallDevEvaluator` ranks each dev anchor against the whole dev
corpus and reports recall@1, delegating to `baselines.metrics.retrieval_metrics`
— the same function the real holdout uses. `primary_metric` is set to
`recall@1`, so `select_best_checkpoint` optimises the target quantity.

Everything else is held identical to Arm A (`abl_a_batch_only_v2`), the current
best nano configuration: same rows, micro-batch 6, accum 2 (effective batch 12 →
245 optimizer steps), lr 2e-4, 5 epochs, `save_total_limit=5`. So any difference
is attributable to the two changes above.

Expected trade, stated in advance so it is not rationalised afterwards: a peakier
softmax is worse-calibrated, so **pair AUC is expected to drop**. That is
acceptable here — this arm exists to test whether recall@1 can be moved at all,
and Arm A's recall@1 gain over frozen nano on all 200 real pairs is currently
+0.0000.
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
        # Arm A's winning corner, held fixed.
        "train_batch_size": 6,
        "eval_batch_size": 6,
        "gradient_accumulation_steps": 2,
        "learning_rate": 2e-4,
        "save_total_limit": 5,
        # The change: select on recall@1, not triplet accuracy.
        "primary_metric": "recall@1",
        "query_prompt": "Represent the query for retrieving supporting documents: ",
        "document_prompt": "Represent the document for retrieval: ",
    },
}

EFFECTIVE_BATCH_TARGET = 12

# Loss knobs — not part of TrainConfig (shared, unmodified), so they travel in
# `extra` and are recorded in run_meta.json for every run.
DEFAULT_LOSS_SCALE = 50.0          # library default is 20.0
DEFAULT_HARDNESS_MODE = "hard_negatives"   # library default is None
DEFAULT_HARDNESS_STRENGTH = 1.0    # library default is 0.0


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
    fields.setdefault("output_dir", Path("artifacts/twotower_top1_optimised") / run_id)
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
            f"{EFFECTIVE_BATCH_TARGET}. This arm will not be comparable to Arm A "
            f"(different optimizer-step count).\n{'!' * 78}\n"
        )
    if cfg.primary_metric != "recall@1":
        print(
            f"\n{'!' * 78}\nWARNING: primary_metric is {cfg.primary_metric!r}, not "
            f"'recall@1' — checkpoint selection will not optimise the target "
            f"metric, which is half the point of this arm.\n{'!' * 78}\n"
        )
    return cfg
