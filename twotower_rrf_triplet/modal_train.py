"""Modal GPU entrypoint for the rrf_003 triplet fine-tune experiment.

Fully separate app/volumes from twotower/modal_train.py (Arm A/B/C) — this
never touches the dorby-twotower-checkpoints volume. Reuses the shared HF
model-download cache volume only (read/write-safe: keyed by model name).

Usage:
  modal run twotower_rrf_triplet/modal_train.py --preset voyage-4-nano --run-id rrf_triplet_voyage_nano_001 --epochs 5
  modal run twotower_rrf_triplet/modal_train.py --preset qwen3-8b --run-id rrf_triplet_qwen3_8b_001 --epochs 5
  modal volume get dorby-twotower-rrf-triplet-checkpoints rrf_triplet_voyage_nano_001 ./artifacts/twotower_rrf_triplet/rrf_triplet_voyage_nano_001
"""

from __future__ import annotations

import modal

APP_NAME = "dorby-twotower-rrf-triplet"
CHECKPOINT_VOLUME = "dorby-twotower-rrf-triplet-checkpoints"
HF_CACHE_VOLUME = "dorby-twotower-hf-cache"  # shared read/write cache, same as twotower/

app = modal.App(APP_NAME)

PRESET_GPU = {
    "voyage-4-nano": "L4",
    "qwen3-8b": "A100-40GB",  # 8B bf16 weights + LoRA; A10G (24GB) OOMs per hf_embedding findings
}

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
    .add_local_python_source("twotower_rrf_triplet", "twotower", "baselines", "synth_pipeline")
    .add_local_dir("data", remote_path="/root/data")
    .add_local_dir("artifacts/twotower_rrf_triplet", remote_path="/root/triplets")
)

checkpoints = modal.Volume.from_name(CHECKPOINT_VOLUME, create_if_missing=True)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True)


@app.function(
    image=image,
    timeout=3 * 60 * 60,
    volumes={
        "/checkpoints": checkpoints,
        "/cache/huggingface": hf_cache,
    },
)
def train_remote(
    preset: str = "voyage-4-nano",
    run_id: str = "rrf_triplet_voyage_nano_001",
    triplets_filename: str = "rrf_003_triplets.json",
    epochs: int = 5,
    seed: int = 42,
    dry_run: bool = False,
    resume_from_checkpoint: str | None = None,
    run_holdout: bool = True,
) -> dict:
    from pathlib import Path

    from twotower_rrf_triplet.config import build_config
    from twotower_rrf_triplet.train import run_training

    triplets_path = Path("/root/triplets") / triplets_filename
    cfg = build_config(
        preset,
        run_id=run_id,
        triplets_path=triplets_path,
        epochs=epochs,
        seed=seed,
        output_dir=Path("/checkpoints") / run_id,
    )
    result = run_training(
        cfg,
        triplets_path,
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
        "preset": preset,
        "adapter_dir": result.get("adapter_dir"),
        "metrics_train_dev": result.get("metrics_train_dev"),
        "metrics_holdout": result.get("metrics_holdout"),
    }


@app.local_entrypoint()
def main(
    preset: str = "voyage-4-nano",
    run_id: str = "rrf_triplet_voyage_nano_001",
    triplets_filename: str = "rrf_003_triplets.json",
    epochs: int = 5,
    seed: int = 42,
    dry_run: bool = False,
    resume_from_checkpoint: str = "",
    run_holdout: bool = True,
    gpu: str = "",
) -> None:
    """CLI: modal run twotower_rrf_triplet/modal_train.py --preset qwen3-8b --run-id rrf_triplet_qwen3_8b_001

    --gpu overrides the preset's default GPU (e.g. --gpu A100-40GB to run
    voyage-4-nano faster than its default L4).
    """
    if preset not in PRESET_GPU:
        raise ValueError(f"unknown preset {preset!r}; choices: {sorted(PRESET_GPU)}")
    call = train_remote.with_options(gpu=gpu or PRESET_GPU[preset])
    result = call.remote(
        preset=preset,
        run_id=run_id,
        triplets_filename=triplets_filename,
        epochs=epochs,
        seed=seed,
        dry_run=dry_run,
        resume_from_checkpoint=resume_from_checkpoint or None,
        run_holdout=run_holdout,
    )
    print("=== Modal rrf-triplet train finished ===")
    for k, v in result.items():
        print(f"{k}: {v}")
    print(
        f"\nPull artifacts with:\n"
        f"  modal volume get {CHECKPOINT_VOLUME} {run_id} ./artifacts/twotower_rrf_triplet/{run_id}"
    )
