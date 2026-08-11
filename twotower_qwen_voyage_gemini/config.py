"""Config: `twotower_qwen_bigbatch`'s winning micro-batch-6 recipe, unchanged,
pointed at the pairing_voyage_gemini rows instead of rrf_003.

Unlike `twotower_qwen_bigbatch`, this package only ever runs the micro-6
corner — that was the clear winner of the two-arm ablation there, so there is
no second (micro-1) arm here to compare against. Every LoRA / optimizer /
prompt field below is copied verbatim from
`twotower_qwen_bigbatch/config.py`'s `qwen3-8b` preset and pinned against it by
`tests/test_qwen_voyage_gemini.py`; only `output_dir`'s package-name segment
and the default `train_batch_size`/`gradient_accumulation_steps` corner (fixed
at micro-6 here, rather than defaulting to the starved micro-1 corner the
upstream preset used) differ.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from twotower.config import TrainConfig

# Qwen3-Embedding-8B's own registered sentence-transformers prompts, copied
# verbatim from twotower_qwen_bigbatch/config.py (which itself copied them
# from twotower_rrf_triplet/config.py), so this experiment's text packing is
# byte-identical to every prior Qwen run's. Query side carries a real
# instruction; document side is intentionally empty (Qwen3 was trained
# asymmetrically, instruction on the query side only).
QWEN3_QUERY_PROMPT = (
    "Instruct: Given a web search query, retrieve relevant passages that "
    "answer the query\nQuery:"
)
QWEN3_DOCUMENT_PROMPT = ""

MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "qwen3-8b": {
        "model_name": "Qwen/Qwen3-Embedding-8B",
        "trust_remote_code": True,
        "truncate_dim": 1024,
        "max_seq_length": 4096,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_target_modules": ("q_proj", "k_proj", "v_proj", "o_proj"),
        # Qwen3-Embedding-8B backbone (Qwen3ForCausalLM) has 36 decoder layers
        # vs. nano's 12 — asserted at training time by add_lora_adapter.
        "expected_layers_per_target": 36,
        # twotower_qwen_bigbatch's measured winner (all-200: pair AUC 0.5947 vs
        # micro-1's 0.5604, MRR 0.3031 vs 0.2734, hard-neg 0.5608 vs 0.4828).
        # Effective batch pinned to 12, same discipline as every ablation this
        # project has run.
        "train_batch_size": 6,
        "eval_batch_size": 6,
        "gradient_accumulation_steps": 2,
        # Qwen's own established rate (twotower_rrf_triplet_qwen3_8b_h100_002),
        # not nano's 2e-4 — unchanged from twotower_qwen_bigbatch.
        "learning_rate": 1e-4,
        # 5 epochs with eval-per-epoch: the default limit of 3 prunes the
        # epoch-1/2 checkpoints before selection runs, which has silently
        # shipped a final-epoch model three separate times in this project
        # (docs/possible-bugs.md #2). Keep all 5.
        "save_total_limit": 5,
        "query_prompt": QWEN3_QUERY_PROMPT,
        "document_prompt": QWEN3_DOCUMENT_PROMPT,
        # Both are load-bearing for fitting micro-batch 6 on an 8B model; see
        # the module docstring.
        "extra": {"torch_dtype": "bfloat16", "gradient_checkpointing_override": True},
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
    """Build a TrainConfig for this experiment; never touches a prior run's dir."""
    if preset not in MODEL_PRESETS:
        raise KeyError(f"unknown preset {preset!r}; choices: {sorted(MODEL_PRESETS)}")

    fields = dict(MODEL_PRESETS[preset])
    extra = dict(fields.get("extra") or {})
    fields.update(overrides)
    # Merge rather than replace, so a caller overriding `extra` cannot silently
    # drop torch_dtype/gradient_checkpointing and OOM at batch 1.
    extra.update(fields.get("extra") or {})
    fields["extra"] = extra

    fields.setdefault("epochs", epochs)
    fields.setdefault("seed", seed)
    fields.setdefault("output_dir", Path("artifacts/twotower_qwen_voyage_gemini") / run_id)
    fields["extra"].update(
        {"rows_path": str(rows_path), "negatives_per_anchor": negatives_per_anchor}
    )

    cfg = TrainConfig(**fields)
    eff = cfg.train_batch_size * cfg.gradient_accumulation_steps
    if eff != EFFECTIVE_BATCH_TARGET:
        print(
            f"\n{'!' * 78}\nWARNING: effective batch is "
            f"{cfg.train_batch_size}x{cfg.gradient_accumulation_steps}={eff}, "
            f"not the target {EFFECTIVE_BATCH_TARGET}. Not comparable to "
            f"twotower_qwen_bigbatch's qwen_micro6_r1.\n{'!' * 78}\n"
        )
    if not cfg.extra.get("gradient_checkpointing_override"):
        print(
            f"\n{'!' * 78}\nWARNING: gradient checkpointing is OFF for an 8B "
            f"model — expect OOM above micro-batch 1.\n{'!' * 78}\n"
        )
    return cfg
