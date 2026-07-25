"""Modal GPU entrypoint for the generic open-weight HF embedding baseline.

Usage:
  modal run baselines/hf_embedding/modal_eval.py --model Qwen/Qwen3-Embedding-8B --holdout-only
  modal run baselines/hf_embedding/modal_eval.py --model BAAI/bge-m3 --holdout-only --run-id bge_m3_holdout
  modal volume get dorby-hf-embedding-eval qwen_qwen3-embedding-8b_holdout ./artifacts/hf_embedding_qwen_qwen3-embedding-8b
"""

from __future__ import annotations

import modal

APP_NAME = "dorby-hf-embedding-eval"
RESULTS_VOLUME = "dorby-hf-embedding-eval"
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
    )
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "TRANSFORMERS_CACHE": "/cache/huggingface",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_python_source("baselines", "synth_pipeline")
    .add_local_dir("data", remote_path="/root/data")
)

results = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=45 * 60,
    volumes={
        "/results": results,
        "/cache/huggingface": hf_cache,
    },
)
def eval_remote(
    run_id: str,
    model: str,
    batch_size: int = 4,
    max_length: int = 8192,
    truncate_dim: int | None = None,
    dtype: str = "auto",
    holdout_only: bool = True,
) -> dict:
    import json
    from pathlib import Path

    from baselines.hf_embedding.eval import run_eval

    artifacts_dir = Path("/results") / run_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    metrics = run_eval(
        data_dir=Path("/root/data"),
        model_name=model,
        batch_size=batch_size,
        max_length=max_length,
        truncate_dim=truncate_dim,
        dtype=dtype,
        artifacts_dir=artifacts_dir,
        holdout_only=holdout_only,
        split_path=Path("/root/data/synthetic/seed_split.json"),
    )
    (artifacts_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    results.commit()
    return {
        "run_id": run_id,
        "model": model,
        "pair": metrics["pair"],
        "retrieval": metrics["retrieval"],
    }


@app.local_entrypoint()
def main(
    model: str,
    run_id: str = "",
    holdout_only: bool = True,
    batch_size: int = 4,
    max_length: int = 8192,
    truncate_dim: int = 0,
    dtype: str = "auto",
    gpu: str = "A10G",
) -> None:
    """CLI: modal run baselines/hf_embedding/modal_eval.py --model Qwen/Qwen3-Embedding-8B --holdout-only"""
    from baselines.hf_embedding.models import slugify

    if not run_id:
        suffix = "holdout" if holdout_only else "full"
        run_id = f"{slugify(model)}_{suffix}"

    call = eval_remote
    if gpu and gpu != "A10G":
        call = eval_remote.with_options(gpu=gpu)

    result = call.remote(
        run_id=run_id,
        model=model,
        batch_size=batch_size,
        max_length=max_length,
        truncate_dim=truncate_dim or None,
        dtype=dtype,
        holdout_only=holdout_only,
    )
    print("=== Modal HF embedding eval finished ===")
    print(f"run_id: {result['run_id']}  model: {result['model']}")
    print("pair:", result["pair"])
    print("retrieval:", result["retrieval"])
    print(
        f"\nPull full metrics with:\n"
        f"  modal volume get {RESULTS_VOLUME} {result['run_id']} "
        f"./artifacts/hf_embedding_{slugify(result['model'])}"
    )
