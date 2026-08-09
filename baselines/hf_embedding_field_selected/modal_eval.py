"""Modal GPU entrypoint for the field-selected HF embedding baseline.

Defaults to A100-40GB, not A10G — A10G (24GB) OOMs on 7-8B models like
Qwen3-Embedding-8B (see docs/hf-embedding-baseline-findings.md and
baselines/voyage_nano_field_selected/modal_eval.py's identical finding).

Usage:
  modal run baselines/hf_embedding_field_selected/modal_eval.py --model Qwen/Qwen3-Embedding-8B --holdout-only
  modal run baselines/hf_embedding_field_selected/modal_eval.py --model Qwen/Qwen3-Embedding-8B
  modal volume get dorby-hf-embedding-field-selected-eval qwen_qwen3-embedding-8b_holdout ./artifacts/hf_embedding_field_selected_modal
"""

from __future__ import annotations

import modal

APP_NAME = "dorby-hf-embedding-field-selected"
RESULTS_VOLUME = "dorby-hf-embedding-field-selected-eval"
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
    .add_local_python_source("baselines", "synth_pipeline")
    .add_local_dir("data", remote_path="/root/data")
)

results = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-40GB",
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
    max_length: int = 4096,
    truncate_dim: int | None = None,
    dtype: str = "auto",
    holdout_only: bool = True,
) -> dict:
    import json
    from pathlib import Path

    from baselines.hf_embedding_field_selected.eval import run_eval

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
    model: str = "Qwen/Qwen3-Embedding-8B",
    run_id: str = "",
    holdout_only: bool = True,
    batch_size: int = 4,
    max_length: int = 4096,
    truncate_dim: int = 0,
    dtype: str = "auto",
    gpu: str = "A100-40GB",
) -> None:
    """CLI: modal run baselines/hf_embedding_field_selected/modal_eval.py --model Qwen/Qwen3-Embedding-8B --holdout-only"""
    from baselines.hf_embedding.models import slugify

    if not run_id:
        suffix = "holdout" if holdout_only else "all"
        run_id = f"{slugify(model)}_{suffix}"

    call = eval_remote
    if gpu and gpu != "A100-40GB":
        call = call.with_options(gpu=gpu)

    result = call.remote(
        run_id=run_id,
        model=model,
        batch_size=batch_size,
        max_length=max_length,
        truncate_dim=truncate_dim or None,
        dtype=dtype,
        holdout_only=holdout_only,
    )
    print("=== Modal field-selected HF embedding eval finished ===")
    print(f"run_id: {result['run_id']}  model: {result['model']}")
    print("pair:", result["pair"])
    print("retrieval:", result["retrieval"])
    print(
        f"\nPull full metrics with:\n"
        f"  modal volume get {RESULTS_VOLUME} {result['run_id']}/metrics.json "
        f"./artifacts/hf_embedding_field_selected_modal/{result['run_id']}/metrics.json"
    )
