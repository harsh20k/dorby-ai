"""Modal GPU entrypoint for the LLM-judge distillation arm.

Usage:
  modal run twotower/modal_train_distill.py --run-id distill_judge_001 --epochs 5
  modal volume get dorby-twotower-checkpoints distill_judge_001 ./artifacts/twotower/distill_judge_001
"""

from __future__ import annotations

import modal

from twotower.modal_train import CHECKPOINT_VOLUME, HF_CACHE_VOLUME, checkpoints, hf_cache, image

APP_NAME = "dorby-twotower-finetune-distill"
app = modal.App(APP_NAME)


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
    run_id: str = "distill_judge_001",
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
    soft_labels_path: str = "/root/data/synthetic/judge_soft_labels_naive.json",
    dry_run: bool = False,
) -> dict:
    from pathlib import Path

    from twotower.config import TrainConfig
    from twotower.train_distill import run_training_distilled

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
        run_holdout=True,
        include_synth=False,  # real-only, matching arm_a_real_only
    )
    result = run_training_distilled(cfg, Path(soft_labels_path), dry_run=dry_run)
    checkpoints.commit()
    hf_cache.commit()
    return {
        "run_id": run_id,
        "output_dir": str(output_dir),
        "split_hash": result.get("split_hash"),
        "data_hash": result.get("data_hash"),
        "distillation": result.get("distillation"),
        "metrics_train_dev": result.get("metrics_train_dev"),
        "metrics_holdout": result.get("metrics_holdout"),
        "adapter_dir": result.get("adapter_dir"),
    }


@app.local_entrypoint()
def main(
    run_id: str = "distill_judge_001",
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
    soft_labels_path: str = "/root/data/synthetic/judge_soft_labels_naive.json",
    dry_run: bool = False,
    gpu: str = "L4",
) -> None:
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
        soft_labels_path=soft_labels_path,
        dry_run=dry_run,
    )
    print("=== Modal distillation train finished ===")
    for k, v in result.items():
        print(f"{k}: {v}")
    print(
        f"\nPull artifacts with:\n"
        f"  modal volume get {CHECKPOINT_VOLUME} {run_id} ./artifacts/twotower/{run_id}"
    )
