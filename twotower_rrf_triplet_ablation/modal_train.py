"""Modal GPU entrypoint for the batch-vs-negatives ablation.

Fully separate app/volumes from twotower/modal_train.py,
twotower_rrf_triplet/modal_train.py, and
twotower_rrf_triplet_bigbatch/modal_train.py — never touches any of their
checkpoint volumes. Reuses the shared HF model-download cache volume only
(read/write-safe: keyed by model name).

All three arms hold train_batch_size * gradient_accumulation_steps = 12, so
every arm runs the identical number of optimizer steps and stays comparable to
the already-completed bigbatch corner. Launch them in parallel:

  modal run --detach twotower_rrf_triplet_ablation/modal_train.py \
      --run-id abl_a_batch_only --rows-filename rrf_003_multineg_k1.json \
      --negatives-per-anchor 1 --train-batch-size 6 --gradient-accumulation-steps 2

  modal run --detach twotower_rrf_triplet_ablation/modal_train.py \
      --run-id abl_b_negs_only --rows-filename rrf_003_multineg_k2.json \
      --negatives-per-anchor 2 --train-batch-size 2 --gradient-accumulation-steps 6

  modal run --detach twotower_rrf_triplet_ablation/modal_train.py \
      --run-id abl_c_baseline --rows-filename rrf_003_multineg_k1.json \
      --negatives-per-anchor 1 --train-batch-size 2 --gradient-accumulation-steps 6

  modal app logs <app-id>   # fetch mode, NOT -f
  modal volume get dorby-twotower-rrf-triplet-ablation-checkpoints <run_id> \
      ./artifacts/twotower_rrf_triplet_ablation/<run_id>
"""

from __future__ import annotations

import modal

APP_NAME = "dorby-twotower-rrf-triplet-ablation"
CHECKPOINT_VOLUME = "dorby-twotower-rrf-triplet-ablation-checkpoints"
HF_CACHE_VOLUME = "dorby-twotower-hf-cache"  # shared read/write cache

app = modal.App(APP_NAME)

# Same GPU class as the bigbatch run so memory behaviour is comparable. The
# probed OOM ceiling for this model at k=2 was micro-batch 8; every arm here
# runs at 6 or 2, and k=1 uses strictly less than k=2.
GPU = "A100-80GB"

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
    .add_local_python_source("twotower_rrf_triplet_ablation", "twotower", "baselines", "synth_pipeline")
    .add_local_dir("data", remote_path="/root/data")
    .add_local_dir("artifacts/twotower_rrf_triplet_ablation", remote_path="/root/rows")
)

checkpoints = modal.Volume.from_name(CHECKPOINT_VOLUME, create_if_missing=True)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU,
    timeout=3 * 60 * 60,
    volumes={
        "/checkpoints": checkpoints,
        "/cache/huggingface": hf_cache,
    },
)
def train_remote(
    run_id: str,
    rows_filename: str,
    negatives_per_anchor: int = 1,
    epochs: int = 5,
    seed: int = 42,
    train_batch_size: int = 2,
    eval_batch_size: int = 6,
    gradient_accumulation_steps: int = 6,
    learning_rate: float = 2e-4,
    dry_run: bool = False,
    resume_from_checkpoint: str | None = None,
    run_holdout: bool = True,
) -> dict:
    from pathlib import Path

    from twotower_rrf_triplet_ablation.config import build_config
    from twotower_rrf_triplet_ablation.train import run_training

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
        "negatives_per_anchor": negatives_per_anchor,
        "train_batch_size": train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "adapter_dir": result.get("adapter_dir"),
        "best_checkpoint": result.get("best_checkpoint"),
        "metrics_train_dev": result.get("metrics_train_dev"),
        "metrics_holdout": result.get("metrics_holdout"),
    }


@app.local_entrypoint()
def main(
    run_id: str = "abl_c_baseline",
    rows_filename: str = "rrf_003_multineg_k1.json",
    negatives_per_anchor: int = 1,
    epochs: int = 5,
    seed: int = 42,
    train_batch_size: int = 2,
    eval_batch_size: int = 6,
    gradient_accumulation_steps: int = 6,
    learning_rate: float = 2e-4,
    dry_run: bool = False,
    resume_from_checkpoint: str = "",
    run_holdout: bool = True,
) -> None:
    """CLI: see module docstring for all three arm invocations."""
    eff = train_batch_size * gradient_accumulation_steps
    if eff != 12:
        print(
            f"WARNING: effective batch {train_batch_size}x{gradient_accumulation_steps}={eff} != 12 "
            "— this arm will not be comparable to the others."
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
    print("=== Modal rrf-triplet-ablation train finished ===")
    for k, v in result.items():
        print(f"{k}: {v}")
    print(
        f"\nPull artifacts with:\n"
        f"  modal volume get {CHECKPOINT_VOLUME} {run_id} ./artifacts/twotower_rrf_triplet_ablation/{run_id}"
    )
