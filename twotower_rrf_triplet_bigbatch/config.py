"""Model preset + config factory for the bigbatch/multi-negative experiment.

Reuses twotower.config.TrainConfig as-is (unmodified, frozen dataclass), same
as twotower_rrf_triplet/config.py did — this module only adds its own preset
values and its own output-path convention
(artifacts/twotower_rrf_triplet_bigbatch/<run_id>), never touching
artifacts/twotower/ or artifacts/twotower_rrf_triplet/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from twotower.config import TrainConfig

# voyage-4-nano only in this experiment (no Qwen3-8B bf16/OOM concerns here).
# train_batch_size/gradient_accumulation_steps are placeholders overridden at
# call time from the empirical Modal batch-size probe result (see
# twotower_rrf_triplet_bigbatch/probe_batch_size.py and the experiment doc for
# the chosen value + why).
MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "voyage-4-nano": {
        "model_name": "voyageai/voyage-4-nano",
        "trust_remote_code": True,
        "truncate_dim": 1024,
        "max_seq_length": 4096,
        "lora_target_modules": ("q_proj", "k_proj", "v_proj", "o_proj"),
        "expected_layers_per_target": 12,
        "train_batch_size": 32,
        "eval_batch_size": 32,
        "gradient_accumulation_steps": 1,
        "learning_rate": 2e-4,
        "query_prompt": "Represent the query for retrieving supporting documents: ",
        "document_prompt": "Represent the document for retrieval: ",
    },
}


def build_config(
    preset: str,
    *,
    run_id: str,
    rows_path: Path,
    negatives_per_anchor: int = 2,
    epochs: int = 5,
    seed: int = 42,
    **overrides: Any,
) -> TrainConfig:
    """Build a TrainConfig for this experiment; never touches twotower/'s or
    twotower_rrf_triplet/'s output dirs."""
    if preset not in MODEL_PRESETS:
        raise KeyError(f"unknown preset {preset!r}; choices: {sorted(MODEL_PRESETS)}")

    fields = dict(MODEL_PRESETS[preset])
    fields.update(overrides)
    fields.setdefault("epochs", epochs)
    fields.setdefault("seed", seed)
    fields.setdefault("output_dir", Path("artifacts/twotower_rrf_triplet_bigbatch") / run_id)
    fields.setdefault("extra", {}).update(
        {"rows_path": str(rows_path), "negatives_per_anchor": negatives_per_anchor}
    )
    return TrainConfig(**fields)
