"""Modal GPU entrypoint for the two-tower LoRA fine-tune.

Usage:
  modal run twotower/modal_train.py --run-id smoke_001 --epochs 1 --max-seq-length 2048
  modal run twotower/modal_train.py --run-id run_001 --epochs 5
  modal volume get dorby-twotower-checkpoints run_001 ./artifacts/twotower/run_001
"""

from __future__ import annotations

import modal

APP_NAME = "dorby-twotower-finetune"
CHECKPOINT_VOLUME = "dorby-twotower-checkpoints"
HF_CACHE_VOLUME = "dorby-twotower-hf-cache"

app = modal.App(APP_NAME)

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
    )
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "TRANSFORMERS_CACHE": "/cache/huggingface",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_python_source("twotower", "baselines", "synth_pipeline")
    .add_local_dir("data", remote_path="/root/data")
)

checkpoints = modal.Volume.from_name(CHECKPOINT_VOLUME, create_if_missing=True)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True)


@app.function(
    image=image,
    gpu="L4",
    timeout=60 * 60,
    volumes={
        "/checkpoints": checkpoints,
        "/cache/huggingface": hf_cache,
    },
)
def train_remote(
    run_id: str = "run_001",
    epochs: int = 5,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    max_seq_length: int = 4096,
    train_batch_size: int = 2,
    eval_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    seed: int = 42,
    truncate_dim: int = 1024,
    run_holdout: bool = False,
    dry_run: bool = False,
    resume_from_checkpoint: str | None = None,
    model_revision: str | None = None,
) -> dict:
    """Run training inside Modal; writes under /checkpoints/<run_id>/."""
    from pathlib import Path

    from twotower.config import TrainConfig
    from twotower.train import run_training

    output_dir = Path("/checkpoints") / run_id
    cfg = TrainConfig(
        data_dir=Path("/root/data"),
        split_path=Path("/root/data/synthetic/seed_split.json"),
        output_dir=output_dir,
        epochs=epochs,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        max_seq_length=max_seq_length,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        seed=seed,
        truncate_dim=truncate_dim,
        run_holdout=run_holdout,
        model_revision=model_revision,
    )
    result = run_training(
        cfg,
        dry_run=dry_run,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    checkpoints.commit()
    hf_cache.commit()
    return {
        "run_id": run_id,
        "output_dir": str(output_dir),
        "split_hash": result.get("split_hash"),
        "data_hash": result.get("data_hash"),
        "metrics_train_dev": result.get("metrics_train_dev"),
        "metrics_holdout": result.get("metrics_holdout"),
        "adapter_dir": result.get("adapter_dir"),
    }


@app.local_entrypoint()
def main(
    run_id: str = "run_001",
    epochs: int = 5,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    max_seq_length: int = 4096,
    train_batch_size: int = 2,
    eval_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    seed: int = 42,
    truncate_dim: int = 1024,
    run_holdout: bool = False,
    dry_run: bool = False,
    resume_from_checkpoint: str = "",
    model_revision: str = "",
    gpu: str = "L4",
) -> None:
    """CLI: modal run twotower/modal_train.py --run-id run_001 --epochs 5"""
    call = train_remote
    if gpu and gpu != "L4":
        call = train_remote.with_options(gpu=gpu)
    result = call.remote(
        run_id=run_id,
        epochs=epochs,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        max_seq_length=max_seq_length,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        seed=seed,
        truncate_dim=truncate_dim,
        run_holdout=run_holdout,
        dry_run=dry_run,
        resume_from_checkpoint=resume_from_checkpoint or None,
        model_revision=model_revision or None,
    )
    print("=== Modal train finished ===")
    for k, v in result.items():
        print(f"{k}: {v}")
    print(
        f"\nPull artifacts with:\n"
        f"  modal volume get {CHECKPOINT_VOLUME} {run_id} ./artifacts/twotower/{run_id}"
    )
