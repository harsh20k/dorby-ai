"""Modal GPU entrypoint: frozen voyage-4-nano on all resolved B-data pairs.

Same packing as queryonly_back_look_001 (seeker=query only,
candidate=background+lookingFor). No LoRA.

Usage:
  modal run --detach bdata_queryonly_back_look_frozen/modal_eval.py --batch-size 8 --gpu A10G
  modal volume get dorby-bdata-queryonly-back-look-frozen allpairs \\
      ./artifacts/bdata_queryonly_back_look_frozen --force
"""

from __future__ import annotations

import modal

APP_NAME = "dorby-bdata-queryonly-back-look-frozen"
RESULTS_VOLUME = "dorby-bdata-queryonly-back-look-frozen"
HF_CACHE_VOLUME = "dorby-twotower-hf-cache"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.51,<5",
        "sentence-transformers>=5.6,<6",
        "accelerate>=0.30",
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
    .add_local_python_source("bdata_queryonly_back_look_frozen", "baselines")
    .add_local_file("data/B-data.json", remote_path="/root/data/B-data.json")
    .add_local_file(
        "data/unique_contacts_B_data.json",
        remote_path="/root/data/unique_contacts_B_data.json",
    )
)

results = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=8 * 60 * 60,
    volumes={
        "/results": results,
        "/cache/huggingface": hf_cache,
    },
)
def eval_remote(
    run_id: str,
    batch_size: int = 8,
    max_length: int = 4096,
    truncate_dim: int = 1024,
    model_name: str = "voyageai/voyage-4-nano",
    retrieval_query_batch_size: int = 256,
) -> dict:
    import json
    from pathlib import Path

    from bdata_queryonly_back_look_frozen.config import ExperimentConfig
    from bdata_queryonly_back_look_frozen.eval import print_bdata_metrics, run_eval

    artifacts_dir = Path("/results") / run_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    cfg = ExperimentConfig(
        source_path=Path("/root/data/B-data.json"),
        unique_contacts_path=Path("/root/data/unique_contacts_B_data.json"),
        artifacts_dir=artifacts_dir,
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
        truncate_dim=truncate_dim,
        retrieval_query_batch_size=retrieval_query_batch_size,
    )
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
        "retrieval": metrics.get("retrieval"),
    }


@app.local_entrypoint()
def main(
    run_id: str = "allpairs",
    batch_size: int = 8,
    max_length: int = 4096,
    truncate_dim: int = 1024,
    model: str = "voyageai/voyage-4-nano",
    gpu: str = "A10G",
    retrieval_query_batch_size: int = 256,
) -> None:
    """CLI: modal run bdata_queryonly_back_look_frozen/modal_eval.py"""
    call = eval_remote
    if gpu and gpu != "A10G":
        call = call.with_options(gpu=gpu)

    result = call.remote(
        run_id=run_id,
        batch_size=batch_size,
        max_length=max_length,
        truncate_dim=truncate_dim,
        model_name=model,
        retrieval_query_batch_size=retrieval_query_batch_size,
    )
    print("=== Modal frozen queryonly_back_look B-data eval finished ===")
    print(f"run_id: {result['run_id']}  model: {result['model']}")
    print("pair:", result["pair"])
    print("retrieval:", result.get("retrieval"))
    print(
        f"\nPull with:\n"
        f"  modal volume get {RESULTS_VOLUME} {result['run_id']} "
        f"./artifacts/bdata_queryonly_back_look_frozen --force"
    )
