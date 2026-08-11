"""Model preset + config factory for the Qwen3-8B micro-batch experiment.

Reuses twotower.config.TrainConfig as-is (unmodified, frozen dataclass), same
convention as the three prior triplet packages. Only adds this experiment's
preset and its own output-path convention
(artifacts/twotower_qwen_bigbatch/<run_id>).

The question
------------
The nano ablation established that **micro-batch size is the lever** that moves
retrieval (Arm A vs Arm C: +0.052 MRR, +0.069 recall@1 at fixed effective
batch). Qwen3-Embedding-8B has the best pair AUC of any model measured in this
project, but its one fine-tune ran at ``train_batch_size=1`` — precisely the
starved setting the ablation showed to be worst — and consequently barely beat
its own frozen baseline (0.6672 vs 0.6595, inside the noise band).

So: does Qwen3-8B improve when given the micro-batch that helped nano?

Why batch 1 was never actually a measured ceiling
-------------------------------------------------
``twotower_rrf_triplet/config.py``'s qwen3-8b preset set ``train_batch_size=1``
citing an OOM at *fp32 on a 40GB A100*. That is a **weights** problem, and it
was already solved in the same preset by ``torch_dtype=bfloat16`` (16GB) plus
``gradient_checkpointing_override=True``. Nobody re-probed the ceiling after
those two fixes landed, on an 80GB card, where the binding constraint is
activations rather than weights. ``probe_batch_size.py`` measures it instead of
assuming it.

Effective batch is pinned to 12
-------------------------------
Same discipline as the nano ablation: ``train_batch_size *
gradient_accumulation_steps == 12`` in every arm, so both arms run the identical
245 optimizer steps on the identical 583 rows and only micro-batch differs.
That makes the Qwen micro-1 vs micro-N comparison read exactly like Arm C vs
Arm A. Learning rate stays at Qwen's own established 1e-4 (not nano's 2e-4) in
both arms, so it is never a second variable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from twotower.config import TrainConfig

# Qwen3-Embedding-8B's own registered sentence-transformers prompts, copied
# verbatim from twotower_rrf_triplet/config.py so this experiment's text
# packing is byte-identical to the prior Qwen run's. Query side carries a real
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
        # Baseline corner (the prior run's starved setting); the big-batch arm
        # overrides both to keep effective batch at 12.
        "train_batch_size": 1,
        "eval_batch_size": 1,
        "gradient_accumulation_steps": 12,
        # Qwen's own established rate from rrf_triplet_qwen3_8b_h100_002, not
        # nano's 2e-4 — held constant across both arms.
        "learning_rate": 1e-4,
        # 5 epochs with eval-per-epoch: the default limit of 3 prunes the
        # epoch-1/2 checkpoints before selection runs, which has silently
        # shipped a final-epoch model three separate times in this project
        # (docs/possible-bugs.md #2). Keep all 5.
        "save_total_limit": 5,
        "query_prompt": QWEN3_QUERY_PROMPT,
        "document_prompt": QWEN3_DOCUMENT_PROMPT,
        # Both are load-bearing for fitting any micro-batch above 1; see the
        # module docstring.
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
    fields.setdefault("output_dir", Path("artifacts/twotower_qwen_bigbatch") / run_id)
    fields["extra"].update(
        {"rows_path": str(rows_path), "negatives_per_anchor": negatives_per_anchor}
    )

    cfg = TrainConfig(**fields)
    eff = cfg.train_batch_size * cfg.gradient_accumulation_steps
    if eff != EFFECTIVE_BATCH_TARGET:
        print(
            f"\n{'!' * 78}\nWARNING: effective batch is "
            f"{cfg.train_batch_size}x{cfg.gradient_accumulation_steps}={eff}, "
            f"not the target {EFFECTIVE_BATCH_TARGET}. Arms will not be "
            f"comparable (optimizer-step counts will differ).\n{'!' * 78}\n"
        )
    if not cfg.extra.get("gradient_checkpointing_override"):
        print(
            f"\n{'!' * 78}\nWARNING: gradient checkpointing is OFF for an 8B "
            f"model — expect OOM above micro-batch 1.\n{'!' * 78}\n"
        )
    return cfg
