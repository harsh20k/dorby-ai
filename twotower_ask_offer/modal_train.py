"""Modal GPU entrypoint: train the two ask/offer towers, then eval on the
real holdout + all-200.

Own app and own checkpoint volume. A100-80GB (matching voyage_gemini_ctrl's
choice) since this run holds two towers + two optimizer states concurrently
and does twice the forward passes per step of a single-tower run.

  modal run --detach twotower_ask_offer/modal_train.py --run-id ask_offer_001

  modal app logs <app-id>   # fetch mode, NOT -f
  modal volume get dorby-twotower-ask-offer-checkpoints ask_offer_001 \\
      ./artifacts/twotower_ask_offer/ask_offer_001

Keep the launching shell alive (`wait`) — `--detach` is still cancelled if
the client process is terminated.
"""

from __future__ import annotations

import modal

APP_NAME = "dorby-twotower-ask-offer"
CHECKPOINT_VOLUME = "dorby-twotower-ask-offer-checkpoints"
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
    .add_local_python_source("twotower_ask_offer", "twotower", "baselines", "eval_real_full", "synth_pipeline")
    .add_local_dir("data", remote_path="/root/data")
    .add_local_dir("eval_real_full/data_frozen", remote_path="/root/eval_real_full/data_frozen")
    # Frozen rows (import_rows.py's output) — never reads pairing_voyage_gemini live.
    .add_local_dir("artifacts/twotower_ask_offer", remote_path="/root/rows")
)

checkpoints = modal.Volume.from_name(CHECKPOINT_VOLUME, create_if_missing=True)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU,
    timeout=4 * 60 * 60,
    volumes={"/checkpoints": checkpoints, "/cache/huggingface": hf_cache},
)
def train_remote(
    run_id: str,
    rows_filename: str = "ask_offer_rows.json",
    lam: float = 1.75,
    epochs: int = 5,
    seed: int = 42,
    dry_run: bool = False,
    run_holdout: bool = True,
    max_rows: int | None = None,
) -> dict:
    from pathlib import Path

    from twotower_ask_offer.config import build_config
    from twotower_ask_offer.train import run_training

    rows_path = Path("/root/rows") / rows_filename
    cfg = build_config(
        run_id=run_id,
        rows_path=rows_path,
        lam=lam,
        epochs=epochs,
        seed=seed,
        output_dir=Path("/checkpoints"),
    )
    result = run_training(
        cfg,
        dry_run=dry_run,
        run_holdout=run_holdout,
        real_holdout_data_dir=Path("/root/data"),
        real_holdout_split_path=Path("/root/data/synthetic/seed_split.json"),
        max_rows=max_rows,
    )
    checkpoints.commit()
    hf_cache.commit()
    return {
        "run_id": run_id,
        "lam": lam,
        "adapter_dir": result.get("adapter_dir"),
        "selection": result.get("selection"),
        "metrics_train_dev": result.get("metrics_train_dev"),
        "metrics_holdout": result.get("metrics_holdout"),
    }


@app.local_entrypoint()
def main(
    run_id: str = "ask_offer_001",
    rows_filename: str = "ask_offer_rows.json",
    lam: float = 1.75,
    epochs: int = 5,
    seed: int = 42,
    dry_run: bool = False,
    run_holdout: bool = True,
    max_rows: int = 0,
) -> None:
    """CLI: see module docstring."""
    result = train_remote.remote(
        run_id=run_id,
        rows_filename=rows_filename,
        lam=lam,
        epochs=epochs,
        seed=seed,
        dry_run=dry_run,
        run_holdout=run_holdout,
        max_rows=(max_rows or None),
    )
    print("=== Modal ask_offer train finished ===")
    for k, v in result.items():
        print(f"{k}: {v}")
    print(
        f"\nPull artifacts with:\n"
        f"  modal volume get {CHECKPOINT_VOLUME} {run_id} "
        f"./artifacts/twotower_ask_offer/{run_id}"
    )
