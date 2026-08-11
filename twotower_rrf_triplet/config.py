"""Model presets + config factory for the rrf_003 triplet fine-tune experiment.

Reuses twotower.config.TrainConfig as-is (unmodified, frozen dataclass) — this
module only adds per-model preset values and an output-path convention that
keeps this experiment's artifacts under artifacts/twotower_rrf_triplet/,
never under artifacts/twotower/ (Arm A/B/C's space).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from twotower.config import TrainConfig

# Qwen3-Embedding-8B's own registered sentence-transformers prompts, pulled
# directly from its HF config_sentence_transformers.json (query prompt is a
# real instruction string; document side is intentionally empty — Qwen3 was
# trained asymmetrically, instruction on the query side only).
QWEN3_QUERY_PROMPT = (
    "Instruct: Given a web search query, retrieve relevant passages that "
    "answer the query\nQuery:"
)
QWEN3_DOCUMENT_PROMPT = ""

MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "voyage-4-nano": {
        "model_name": "voyageai/voyage-4-nano",
        "trust_remote_code": True,
        "truncate_dim": 1024,
        "max_seq_length": 4096,
        "lora_target_modules": ("q_proj", "k_proj", "v_proj", "o_proj"),
        "expected_layers_per_target": 12,
        "train_batch_size": 2,
        "eval_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        # Voyage's own asymmetric convention, matching twotower/config.py's
        # defaults (kept explicit here since this module never imports those
        # defaults, by design — this file's presets are the single source of
        # truth for this experiment).
        "query_prompt": "Represent the query for retrieving supporting documents: ",
        "document_prompt": "Represent the document for retrieval: ",
    },
    "qwen3-8b": {
        "model_name": "Qwen/Qwen3-Embedding-8B",
        "trust_remote_code": True,
        "truncate_dim": 1024,
        "max_seq_length": 4096,
        "lora_target_modules": ("q_proj", "k_proj", "v_proj", "o_proj"),
        # Qwen3-Embedding-8B backbone (Qwen3ForCausalLM) has 36 decoder layers
        # (confirmed via its HF config.json num_hidden_layers), vs. nano's 12.
        "expected_layers_per_target": 36,
        # 8B params + LoRA + long sequences needs a much smaller per-device
        # batch than nano; compensate with more accumulation steps to keep a
        # similar effective batch size.
        "train_batch_size": 1,
        "eval_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 1e-4,
        "query_prompt": QWEN3_QUERY_PROMPT,
        "document_prompt": QWEN3_DOCUMENT_PROMPT,
        # fp32 load of an 8B model OOMs even batch_size=1 on a 40GB A100
        # (confirmed: 39.4GB used of 39.49GB, tried to allocate 96MiB more) —
        # bf16 halves persistent weight memory, same as the already-proven
        # baselines/hf_embedding path for this model family. See
        # twotower_rrf_triplet/model.py::build_model_with_dtype.
        "extra": {"torch_dtype": "bfloat16", "gradient_checkpointing_override": True},
    },
}


def build_config(
    preset: str,
    *,
    run_id: str,
    triplets_path: Path,
    epochs: int = 5,
    seed: int = 42,
    **overrides: Any,
) -> TrainConfig:
    """Build a TrainConfig for this experiment; never touches twotower/'s defaults dir."""
    if preset not in MODEL_PRESETS:
        raise KeyError(f"unknown preset {preset!r}; choices: {sorted(MODEL_PRESETS)}")

    fields = dict(MODEL_PRESETS[preset])
    fields.update(overrides)
    fields.setdefault("epochs", epochs)
    fields.setdefault("seed", seed)
    fields.setdefault("output_dir", Path("artifacts/twotower_rrf_triplet") / run_id)
    # Not used by this training path (no build_split_bundle call), kept only
    # so TrainConfig.to_dict() round-trips cleanly for run_meta.json.
    fields.setdefault("extra", {}).update({"triplets_path": str(triplets_path)})
    return TrainConfig(**fields)
