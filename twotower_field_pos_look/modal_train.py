"""Modal GPU entrypoint: LoRA fine-tune on positioning+lookingFor-only text.

Own app and own checkpoint volume — never touches any prior experiment's
volume. Mounts this package's own row file (built by
`scripts/build_rrf_multineg_triplets_pos_look.py` from the same 643-row
`rrf_003` population `top1_ctrl` trained on, just with trimmed text) — not
`top1_ctrl`'s row file, since the text differs.

  modal run --detach twotower_field_pos_look/modal_train.py --run-id field_pos_look_001

  modal app logs <app-id>   # fetch mode, NOT -f
  modal volume get dorby-twotower-field-pos-look-checkpoints <run_id> \\
      ./artifacts/twotower_field_pos_look/<run_id>

Keep the launching shell alive (`wait`) — `--detach` is still cancelled if
the client process is terminated.
"""

from __future__ import annotations

import modal

APP_NAME = "dorby-twotower-field-pos-look"
CHECKPOINT_VOLUME = "dorby-twotower-field-pos-look-checkpoints"
HF_CACHE_VOLUME = "dorby-twotower-hf-cache"  # shared read/write cache

app = modal.App(APP_NAME)

GPU = "A100-80GB"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.51,<5",
        "sentence-transformers>=5.6,<6",
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
    .add_local_python_source("twotower_field_pos_look", "twotower", "baselines", "synth_pipeline")
    .add_local_dir("data", remote_path="/root/data")
    # This package's own row file (trimmed text), not top1_ctrl's.
    .add_local_dir("artifacts/twotower_field_pos_look", remote_path="/root/rows")
)

checkpoints = modal.Volume.from_name(CHECKPOINT_VOLUME, create_if_missing=True)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU,
    timeout=3 * 60 * 60,
    volumes={"/checkpoints": checkpoints, "/cache/huggingface": hf_cache},
)
def train_remote(
    run_id: str,
    rows_filename: str = "rrf_003_multineg_k1_pos_look.json",
    negatives_per_anchor: int = 1,
    epochs: int = 5,
    seed: int = 42,
    train_batch_size: int = 6,
    eval_batch_size: int = 6,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 2e-4,
    loss_scale: float = 20.0,
    dry_run: bool = False,
    resume_from_checkpoint: str | None = None,
) -> dict:
    from pathlib import Path

    from twotower_field_pos_look.config import build_config
    from twotower_field_pos_look.train import run_training

    rows_path = Path("/root/rows") / rows_filename
    cfg = build_config(
        "voyage-4-nano",
        run_id=run_id,
        rows_path=rows_path,
        negatives_per_anchor=negatives_per_anchor,
        epochs=epochs,
        seed=seed,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        loss_scale=loss_scale,
        output_dir=Path("/checkpoints") / run_id,
    )
    result = run_training(
        cfg,
        rows_path,
        negatives_per_anchor=negatives_per_anchor,
        dry_run=dry_run,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    checkpoints.commit()
    hf_cache.commit()
    return {
        "run_id": run_id,
        "loss_scale": loss_scale,
        "adapter_dir": result.get("adapter_dir"),
        "best_checkpoint": result.get("best_checkpoint"),
        "metrics_train_dev": result.get("metrics_train_dev"),
    }


@app.local_entrypoint()
def main(
    run_id: str = "field_pos_look_001",
    rows_filename: str = "rrf_003_multineg_k1_pos_look.json",
    negatives_per_anchor: int = 1,
    epochs: int = 5,
    seed: int = 42,
    train_batch_size: int = 6,
    eval_batch_size: int = 6,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 2e-4,
    loss_scale: float = 20.0,
    dry_run: bool = False,
    resume_from_checkpoint: str = "",
) -> None:
    """CLI: see module docstring."""
    eff = train_batch_size * gradient_accumulation_steps
    if eff != 12:
        print(
            f"WARNING: effective batch {train_batch_size}x{gradient_accumulation_steps}={eff} "
            "!= 12 — not comparable to top1_ctrl."
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
        loss_scale=loss_scale,
        dry_run=dry_run,
        resume_from_checkpoint=resume_from_checkpoint or None,
    )
    print("=== Modal field_pos_look train finished ===")
    for k, v in result.items():
        print(f"{k}: {v}")
    print(
        f"\nPull artifacts with:\n"
        f"  modal volume get {CHECKPOINT_VOLUME} {run_id} "
        f"./artifacts/twotower_field_pos_look/{run_id}"
    )
