"""Modal GPU entrypoint for B-data Voyage-4-nano holdout eval.

Usage:
  modal run bdata_voyage_nano/modal_eval.py
  modal run bdata_voyage_nano/modal_eval.py --batch-size 8 --gpu A10G
  modal volume get dorby-bdata-voyage-nano-eval holdout ./artifacts/bdata_voyage_nano
"""

from __future__ import annotations

import modal

APP_NAME = "dorby-bdata-voyage-nano-eval"
RESULTS_VOLUME = "dorby-bdata-voyage-nano-eval"
HF_CACHE_VOLUME = "dorby-twotower-hf-cache"  # reuse existing HF model cache

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.51,<5",
        "sentence-transformers>=3.4.1,<6",
        "accelerate>=0.30",
        "scikit-learn>=1.4.0",
        "numpy>=1.26.0",
        "tqdm>=4.66.0",
    )
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "TRANSFORMERS_CACHE": "/cache/huggingface",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_python_source("bdata_voyage_nano", "baselines")
    .add_local_file("data/B-data.json", remote_path="/root/data/B-data.json")
    .add_local_file(
        "bdata_voyage_nano/split.json",
        remote_path="/root/bdata_voyage_nano/split.json",
    )
)

results = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=4 * 60 * 60,
    volumes={
        "/results": results,
        "/cache/huggingface": hf_cache,
    },
)
def eval_remote(
    run_id: str,
    batch_size: int = 4,
    max_length: int = 8192,
    truncate_dim: int = 1024,
    model_name: str = "voyageai/voyage-4-nano",
) -> dict:
    import json
    from pathlib import Path

    from bdata_voyage_nano.config import ExperimentConfig
    from bdata_voyage_nano.eval import print_bdata_metrics, run_eval

    artifacts_dir = Path("/results") / run_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # On Modal the package lives under /root; point source + split at the
    # files baked into the image. Artifacts go to the results volume.
    cfg = ExperimentConfig(
        source_path=Path("/root/data/B-data.json"),
        split_path=Path("/root/bdata_voyage_nano/split.json"),
        artifacts_dir=artifacts_dir,
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
        truncate_dim=truncate_dim,
    )

    # Source lock check: Modal copy is a normal file (writable). Bypass by
    # chmod'ing read-only so assert_source_locked still holds.
    cfg.source_path.chmod(0o444)

    metrics = run_eval(cfg)
    print_bdata_metrics(metrics)
    (artifacts_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    results.commit()
    return {
        "run_id": run_id,
        "model": model_name,
        "pair": metrics["pair"],
        "within_seeker": metrics.get("within_seeker"),
        "slices": metrics.get("slices"),
    }


@app.local_entrypoint()
def main(
    run_id: str = "holdout",
    batch_size: int = 8,
    max_length: int = 8192,
    truncate_dim: int = 1024,
    model: str = "voyageai/voyage-4-nano",
    gpu: str = "A10G",
) -> None:
    """CLI: modal run bdata_voyage_nano/modal_eval.py"""
    call = eval_remote
    if gpu and gpu != "A10G":
        call = call.with_options(gpu=gpu)

    result = call.remote(
        run_id=run_id,
        batch_size=batch_size,
        max_length=max_length,
        truncate_dim=truncate_dim,
        model_name=model,
    )
    print("=== Modal B-data Voyage-4-nano eval finished ===")
    print(f"run_id: {result['run_id']}  model: {result['model']}")
    print("pair:", result["pair"])
    print(
        f"\nPull full metrics + embedding caches with:\n"
        f"  modal volume get {RESULTS_VOLUME} {result['run_id']} "
        f"./artifacts/bdata_voyage_nano"
    )
