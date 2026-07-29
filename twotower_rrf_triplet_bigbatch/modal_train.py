"""Modal GPU entrypoint for the bigbatch/multi-negative rrf_003 fine-tune.

Fully separate app/volumes from both twotower/modal_train.py (Arm A/B/C) and
twotower_rrf_triplet/modal_train.py (the first triplet fine-tune) — this
never touches either of those checkpoint volumes. Reuses the shared HF
model-download cache volume only (read/write-safe: keyed by model name).

Usage:
  # empirical batch-size probe first (see probe_batch_size.py)
  modal run twotower_rrf_triplet_bigbatch/probe_batch_size.py

  modal run twotower_rrf_triplet_bigbatch/modal_train.py --detach \
      --run-id rrf_triplet_voyage_nano_bigbatch_001 \
      --train-batch-size 32 --negatives-per-anchor 2 --epochs 5
  modal app logs <app-id>   # fetch mode, NOT -f — see docs/twotower-rrf-triplet-experiment.md
  modal volume get dorby-twotower-rrf-triplet-bigbatch-checkpoints rrf_triplet_voyage_nano_bigbatch_001 \
      ./artifacts/twotower_rrf_triplet_bigbatch/rrf_triplet_voyage_nano_bigbatch_001
"""

from __future__ import annotations

import modal

APP_NAME = "dorby-twotower-rrf-triplet-bigbatch"
CHECKPOINT_VOLUME = "dorby-twotower-rrf-triplet-bigbatch-checkpoints"
HF_CACHE_VOLUME = "dorby-twotower-hf-cache"  # shared read/write cache, same as twotower/ and twotower_rrf_triplet/

app = modal.App(APP_NAME)

# voyage-4-nano is small (347M params); rrf_triplet_voyage_nano_001 already
# ran on an 80GB A100 at batch_size=2 with large headroom, so this experiment
# stays on the same GPU class rather than the smaller default L4.
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
    .add_local_python_source("twotower_rrf_triplet_bigbatch", "twotower", "baselines", "synth_pipeline")
    .add_local_dir("data", remote_path="/root/data")
    .add_local_dir("artifacts/twotower_rrf_triplet_bigbatch", remote_path="/root/rows")
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
    run_id: str = "rrf_triplet_voyage_nano_bigbatch_001",
    rows_filename: str = "rrf_003_multineg_k2.json",
    negatives_per_anchor: int = 2,
    epochs: int = 5,
    seed: int = 42,
    train_batch_size: int = 32,
    eval_batch_size: int = 32,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 2e-4,
    dry_run: bool = False,
    resume_from_checkpoint: str | None = None,
    run_holdout: bool = True,
) -> dict:
    from pathlib import Path

    from twotower_rrf_triplet_bigbatch.config import build_config
    from twotower_rrf_triplet_bigbatch.train import run_training

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
        "adapter_dir": result.get("adapter_dir"),
        "metrics_train_dev": result.get("metrics_train_dev"),
        "metrics_holdout": result.get("metrics_holdout"),
    }


@app.local_entrypoint()
def main(
    run_id: str = "rrf_triplet_voyage_nano_bigbatch_001",
    rows_filename: str = "rrf_003_multineg_k2.json",
    negatives_per_anchor: int = 2,
    epochs: int = 5,
    seed: int = 42,
    train_batch_size: int = 32,
    eval_batch_size: int = 32,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 2e-4,
    dry_run: bool = False,
    resume_from_checkpoint: str = "",
    run_holdout: bool = True,
) -> None:
    """CLI: modal run twotower_rrf_triplet_bigbatch/modal_train.py --detach \
        --run-id rrf_triplet_voyage_nano_bigbatch_001 --train-batch-size 32
    """
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
    print("=== Modal rrf-triplet-bigbatch train finished ===")
    for k, v in result.items():
        print(f"{k}: {v}")
    print(
        f"\nPull artifacts with:\n"
        f"  modal volume get {CHECKPOINT_VOLUME} {run_id} ./artifacts/twotower_rrf_triplet_bigbatch/{run_id}"
    )
