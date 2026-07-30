"""Modal GPU entrypoint for the Qwen3-8B micro-batch experiment.

Own app and own checkpoint volume — never touches twotower/,
twotower_rrf_triplet/, twotower_rrf_triplet_bigbatch/, or
twotower_rrf_triplet_ablation/ volumes. Shares only the HF model-download cache
(read/write-safe: keyed by model name), which already holds Qwen3-Embedding-8B
from the prior run.

Both arms hold train_batch_size * gradient_accumulation_steps = 12, so each runs
the identical 245 optimizer steps and only micro-batch differs — the same design
that made the nano ablation readable.

  # probe the real ceiling first (batch 1 was never measured, see config.py)
  modal run twotower_qwen_bigbatch/probe_batch_size.py

  # big-batch arm (micro-batch from the probe; 6 shown)
  modal run --detach twotower_qwen_bigbatch/modal_train.py \\
      --run-id qwen_micro6 --train-batch-size 6 --gradient-accumulation-steps 2

  # control arm — the prior run's starved setting, at matched effective batch
  modal run --detach twotower_qwen_bigbatch/modal_train.py \\
      --run-id qwen_micro1 --train-batch-size 1 --gradient-accumulation-steps 12

  modal app logs <app-id>   # fetch mode, NOT -f
  modal volume get dorby-twotower-qwen-bigbatch-checkpoints <run_id> \\
      ./artifacts/twotower_qwen_bigbatch/<run_id>
"""

from __future__ import annotations

import modal

APP_NAME = "dorby-twotower-qwen-bigbatch"
CHECKPOINT_VOLUME = "dorby-twotower-qwen-bigbatch-checkpoints"
HF_CACHE_VOLUME = "dorby-twotower-hf-cache"  # shared read/write cache

app = modal.App(APP_NAME)

# 8B params in bf16 (~16GB resident) plus activations for a real micro-batch.
# The A100-80GB the nano ablation used is the floor here, not the ceiling —
# override with --gpu H200 if the probe says 80GB is binding.
GPU = "H100"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.51,<5",
        "sentence-transformers>=3.4.1,<6",
        "peft>=0.14,<1",
        "accelerate>=0.30",
        "datasets>=2.19",
        "scikit-learn>=1.4.0",
        "numpy>=1.26.0",
        "tqdm>=4.66.0",
        "python-dotenv>=1.0.0",
        "einops",
    )
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "TRANSFORMERS_CACHE": "/cache/huggingface",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_python_source(
        "twotower_qwen_bigbatch", "twotower", "baselines", "synth_pipeline"
    )
    .add_local_dir("data", remote_path="/root/data")
    # Reuses the ablation's row files read-only: identical inputs mean the Qwen
    # arms train on exactly the population Arm A did, so the two experiments'
    # micro-batch effects are directly comparable.
    .add_local_dir("artifacts/twotower_rrf_triplet_ablation", remote_path="/root/rows")
)

checkpoints = modal.Volume.from_name(CHECKPOINT_VOLUME, create_if_missing=True)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU,
    timeout=8 * 60 * 60,
    volumes={"/checkpoints": checkpoints, "/cache/huggingface": hf_cache},
)
def train_remote(
    run_id: str,
    rows_filename: str = "rrf_003_multineg_k1.json",
    negatives_per_anchor: int = 1,
    epochs: int = 5,
    seed: int = 42,
    train_batch_size: int = 1,
    eval_batch_size: int = 1,
    gradient_accumulation_steps: int = 12,
    learning_rate: float = 1e-4,
    dry_run: bool = False,
    resume_from_checkpoint: str | None = None,
    run_holdout: bool = True,
) -> dict:
    from pathlib import Path

    from twotower_qwen_bigbatch.config import build_config
    from twotower_qwen_bigbatch.train import run_training

    rows_path = Path("/root/rows") / rows_filename
    cfg = build_config(
        "qwen3-8b",
        run_id=run_id,
        rows_path=rows_path,
        negatives_per_anchor=negatives_per_anchor,
        epochs=epochs,
        seed=seed,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        output_dir=Path("/checkpoints") / run_id,
    )
    result = run_training(
        cfg,
        rows_path,
        negatives_per_anchor=negatives_per_anchor,
        dry_run=dry_run,
        run_holdout=run_holdout,
        real_holdout_data_dir=Path("/root/data"),
        real_holdout_split_path=Path("/root/data/synthetic/seed_split.json"),
        resume_from_checkpoint=resume_from_checkpoint,
    )
    checkpoints.commit()
    hf_cache.commit()
    return {
        "run_id": run_id,
        "train_batch_size": train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "adapter_dir": result.get("adapter_dir"),
        "best_checkpoint": result.get("best_checkpoint"),
        "metrics_train_dev": result.get("metrics_train_dev"),
        "metrics_holdout": result.get("metrics_holdout"),
    }


@app.local_entrypoint()
def main(
    run_id: str = "qwen_micro1",
    rows_filename: str = "rrf_003_multineg_k1.json",
    negatives_per_anchor: int = 1,
    epochs: int = 5,
    seed: int = 42,
    train_batch_size: int = 1,
    eval_batch_size: int = 1,
    gradient_accumulation_steps: int = 12,
    learning_rate: float = 1e-4,
    dry_run: bool = False,
    resume_from_checkpoint: str = "",
    run_holdout: bool = True,
) -> None:
    """CLI: see module docstring for both arm invocations."""
    eff = train_batch_size * gradient_accumulation_steps
    if eff != 12:
        print(
            f"WARNING: effective batch {train_batch_size}x{gradient_accumulation_steps}={eff} "
            "!= 12 — this arm will not be comparable to the other."
        )
    result = train_remote.remote(
        run_id=run_id,
        rows_filename=rows_filename,
        negatives_per_anchor=negatives_per_anchor,
        epochs=epochs,
        seed=seed,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        dry_run=dry_run,
        resume_from_checkpoint=resume_from_checkpoint or None,
        run_holdout=run_holdout,
    )
    print("=== Modal qwen-bigbatch train finished ===")
    for k, v in result.items():
        print(f"{k}: {v}")
    print(
        f"\nPull artifacts with:\n"
        f"  modal volume get {CHECKPOINT_VOLUME} {run_id} "
        f"./artifacts/twotower_qwen_bigbatch/{run_id}"
    )
