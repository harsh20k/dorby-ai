"""Modal GPU entrypoint: train the split two-tower (separate query/candidate
LoRA adapters), reusing twotower_query_only/'s already-built rows.

Own app and checkpoint volume. Loads two copies of voyage-4-nano (one per
tower) — well within an A100-80GB's memory even with optimizer state for
both.

    modal run --detach twotower_split/modal_train.py --run-id split_001
    modal volume get dorby-twotower-split-checkpoints split_001 \\
        ./artifacts/twotower_split/split_001
"""

from __future__ import annotations

import modal

APP_NAME = "dorby-twotower-split"
CHECKPOINT_VOLUME = "dorby-twotower-split-checkpoints"
HF_CACHE_VOLUME = "dorby-twotower-hf-cache"

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
        "einops",
    )
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "TRANSFORMERS_CACHE": "/cache/huggingface",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_python_source("twotower_split", "twotower", "baselines", "synth_pipeline")
    .add_local_dir("data", remote_path="/root/data")
    .add_local_dir("artifacts/twotower_query_only", remote_path="/root/rows")
)

checkpoints = modal.Volume.from_name(CHECKPOINT_VOLUME, create_if_missing=True)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU,
    timeout=3 * 60 * 60,
    volumes={"/checkpoints": checkpoints, "/cache/huggingface": hf_cache},
)
def train_remote(run_id: str, epochs: int = 5, train_batch_size: int = 6, dry_run: bool = False) -> dict:
    from pathlib import Path

    from twotower_split.config import build_config
    from twotower_split.train import run_training

    cfg = build_config(
        run_id=run_id, epochs=epochs, train_batch_size=train_batch_size,
        output_dir=Path("/checkpoints") / run_id,
    )
    rows_path = Path("/root/rows/rrf_003_multineg_k1_query_only.json")
    result = run_training(cfg, rows_path, dry_run=dry_run)
    checkpoints.commit()
    hf_cache.commit()
    return result


@app.local_entrypoint()
def main(run_id: str = "split_001", epochs: int = 5, train_batch_size: int = 6, dry_run: bool = False) -> None:
    result = train_remote.remote(run_id=run_id, epochs=epochs, train_batch_size=train_batch_size, dry_run=dry_run)
    print("=== Modal split two-tower train finished ===")
    for k, v in result.items():
        if k not in ("loss_history",):
            print(f"{k}: {v}")
    print(
        f"\nPull artifacts with:\n"
        f"  modal volume get {CHECKPOINT_VOLUME} {run_id} ./artifacts/twotower_split/{run_id}"
    )
