"""Modal GPU entrypoint for the lambda sensitivity sweep (no fitting step).

Thin wrapper around baselines.reciprocal_lambda_grid.eval.run_eval — same
code, same metrics, run on a Modal GPU instead of local MPS. Mirrors
baselines/reciprocal_static/modal_eval.py's shape.

Usage:
  modal run baselines/reciprocal_lambda_grid/modal_eval.py
  modal volume get dorby-reciprocal-lambda-grid-eval real200 ./artifacts/reciprocal_lambda_grid
"""

from __future__ import annotations

import modal

APP_NAME = "dorby-reciprocal-lambda-grid-eval"
RESULTS_VOLUME = "dorby-reciprocal-lambda-grid-eval"
HF_CACHE_VOLUME = "dorby-twotower-hf-cache"  # reuse the existing HF model cache

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
        "datasets>=2.19",
        "einops",
    )
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "TRANSFORMERS_CACHE": "/cache/huggingface",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_python_source("baselines", "eval_real_full", "twotower", "synth_pipeline")
    .add_local_dir("data", remote_path="/root/data")
    .add_local_dir("eval_real_full/data_frozen", remote_path="/root/eval_real_full/data_frozen")
)

results = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=30 * 60,
    volumes={"/results": results, "/cache/huggingface": hf_cache},
)
def eval_remote(
    run_id: str,
    model: str = "voyageai/voyage-4-nano",
    batch_size: int = 8,
    max_length: int = 8192,
    truncate_dim: int | None = 1024,
    lambda_min: float = -2.0,
    lambda_max: float = 2.0,
    lambda_step: float = 0.05,
) -> dict:
    import json
    from pathlib import Path

    from baselines.reciprocal_lambda_grid.eval import run_eval
    from baselines.reciprocal_static.eval import build_lambda_grid

    artifacts_dir = Path("/results") / run_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    lambda_grid = build_lambda_grid(lambda_min, lambda_max, lambda_step)
    metrics = run_eval(
        data_dir=Path("/root/data"),
        split_path=Path("/root/data/synthetic/seed_split.json"),
        model_name=model,
        batch_size=batch_size,
        max_length=max_length,
        truncate_dim=truncate_dim,
        artifacts_dir=artifacts_dir,
        lambda_grid=lambda_grid,
    )
    (artifacts_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    results.commit()
    return metrics


@app.local_entrypoint()
def main(
    run_id: str = "real200",
    model: str = "voyageai/voyage-4-nano",
    batch_size: int = 8,
    max_length: int = 8192,
    truncate_dim: int = 1024,
    lambda_min: float = -2.0,
    lambda_max: float = 2.0,
    lambda_step: float = 0.05,
    gpu: str = "A10G",
) -> None:
    """CLI: modal run baselines/reciprocal_lambda_grid/modal_eval.py"""
    call = eval_remote
    if gpu and gpu != "A10G":
        call = call.with_options(gpu=gpu)

    metrics = call.remote(
        run_id=run_id,
        model=model,
        batch_size=batch_size,
        max_length=max_length,
        truncate_dim=truncate_dim or None,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        lambda_step=lambda_step,
    )
    print("=== Modal reciprocal-lambda-grid eval finished ===")
    for name in ("holdout", "all"):
        s = metrics["subsets"][name]
        print(
            f"{name}: forward-only AUC={s['forward_only_auc']:.4f} "
            f"curve max=(lambda={s['best_lambda']:.2f}, AUC={s['best_auc']:.4f})"
        )
    print(
        f"\nPull full metrics with:\n  modal volume get {RESULTS_VOLUME} {run_id} "
        f"./artifacts/reciprocal_lambda_grid_{run_id}"
    )
