"""Modal GPU entrypoint for the lookingFor-sectioning experiment.

Usage:
  modal run baselines/voyage_nano_sectioned/modal_eval.py --holdout-only
  modal run baselines/voyage_nano_sectioned/modal_eval.py --holdout-only --direction seeker
  modal run baselines/voyage_nano_sectioned/modal_eval.py --user-id cm5yrbzzj01uhui0ly2n8tqpi
  modal volume get dorby-sectioning-eval holdout_candidate ./artifacts/voyage_nano_sectioned_modal/holdout_candidate
"""

from __future__ import annotations

import modal

APP_NAME = "dorby-lookingfor-sectioning"
RESULTS_VOLUME = "dorby-sectioning-eval"
HF_CACHE_VOLUME = "dorby-twotower-hf-cache"  # reuse the existing HF model cache

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.51,<5",
        "sentence-transformers>=3.4.1,<6",
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
    gpu="L4",
    timeout=30 * 60,
    volumes={
        "/results": results,
        "/cache/huggingface": hf_cache,
    },
)
def eval_remote(
    run_id: str = "holdout_candidate",
    direction: str = "candidate",
    holdout_only: bool = True,
    user_id: str | None = None,
    model: str = "voyageai/voyage-4-nano",
    batch_size: int = 16,
    max_length: int = 8192,
    truncate_dim: int = 1024,
    agg: str = "max",
    topk: int = 2,
    temperature: float = 0.05,
) -> dict:
    """Run eval.py (direction="candidate") or eval_seeker.py ("seeker") remotely.

    ``agg``/``topk``/``temperature`` are only meaningful for direction="seeker"
    (see baselines/voyage_nano_sectioned/aggregate.py); eval.py (candidate
    direction) doesn't accept them.
    """
    import json
    from pathlib import Path

    if direction == "candidate":
        from baselines.voyage_nano_sectioned.eval import run_eval
    elif direction == "seeker":
        from baselines.voyage_nano_sectioned.eval_seeker import run_eval
    else:
        raise ValueError(f"direction must be 'candidate' or 'seeker', got {direction!r}")

    artifacts_dir = Path("/results") / run_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    extra_kwargs = {}
    if direction == "seeker":
        extra_kwargs = {"agg": agg, "topk": topk, "temperature": temperature}

    metrics = run_eval(
        data_dir=Path("/root/data"),
        model_name=model,
        batch_size=batch_size,
        max_length=max_length,
        truncate_dim=truncate_dim,
        artifacts_dir=artifacts_dir,
        holdout_only=holdout_only,
        split_path=Path("/root/data/synthetic/seed_split.json"),
        user_id=user_id,
        **extra_kwargs,
    )
    (artifacts_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    results.commit()
    return {
        "run_id": run_id,
        "direction": direction,
        "agg": metrics.get("agg"),
        "pair": metrics["pair"],
        "retrieval": metrics["retrieval"],
    }


@app.function(
    image=image,
    gpu="L4",
    timeout=30 * 60,
    volumes={
        "/results": results,
        "/cache/huggingface": hf_cache,
    },
)
def eval_remote_seeker_multi_agg(
    run_id: str = "holdout_seeker_multi_agg",
    holdout_only: bool = True,
    user_id: str | None = None,
    model: str = "voyageai/voyage-4-nano",
    batch_size: int = 16,
    max_length: int = 8192,
    truncate_dim: int = 1024,
    topk: int = 2,
    temperature: float = 0.05,
) -> dict:
    """Encode sections/candidates once (direction='seeker'), score topk_mean + softmax
    from that single embedding pass — one GPU invocation covers both new agg modes.
    """
    import json
    from pathlib import Path

    from baselines.voyage_nano_sectioned.eval_seeker import run_eval_multi_agg

    artifacts_dir = Path("/results") / run_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    agg_configs = [
        {"label": "topk_mean", "agg": "topk_mean", "topk": topk, "temperature": temperature},
        {"label": "softmax", "agg": "softmax", "topk": topk, "temperature": temperature},
    ]
    all_metrics = run_eval_multi_agg(
        data_dir=Path("/root/data"),
        model_name=model,
        batch_size=batch_size,
        max_length=max_length,
        truncate_dim=truncate_dim,
        artifacts_dir=artifacts_dir,
        agg_configs=agg_configs,
        holdout_only=holdout_only,
        split_path=Path("/root/data/synthetic/seed_split.json"),
        user_id=user_id,
    )
    (artifacts_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2) + "\n")
    results.commit()
    return {
        "run_id": run_id,
        "direction": "seeker",
        "results": {
            label: {"pair": m["pair"], "retrieval": m["retrieval"]}
            for label, m in all_metrics.items()
        },
    }


@app.local_entrypoint()
def multi_agg(
    run_id: str = "",
    holdout_only: bool = True,
    user_id: str = "",
    model: str = "voyageai/voyage-4-nano",
    batch_size: int = 4,
    max_length: int = 4096,
    truncate_dim: int = 1024,
    gpu: str = "A100-40GB",
    topk: int = 2,
    temperature: float = 0.05,
) -> None:
    """CLI: modal run baselines/voyage_nano_sectioned/modal_eval.py::multi_agg --holdout-only

    Single combined Modal invocation: encodes seeker sections + candidate corpus
    once, then scores both topk_mean and softmax aggregation modes from that one
    pass (see run_eval_multi_agg). Defaults match the sectioning-seeker-agg
    experiment: batch_size=4, max_length=4096, A100-40GB.
    """
    if not run_id:
        run_id = "holdout_seeker_multi_agg"

    call = eval_remote_seeker_multi_agg
    if gpu and gpu != "L4":
        call = eval_remote_seeker_multi_agg.with_options(gpu=gpu)

    result = call.remote(
        run_id=run_id,
        holdout_only=holdout_only,
        user_id=user_id or None,
        model=model,
        batch_size=batch_size,
        max_length=max_length,
        truncate_dim=truncate_dim,
        topk=topk,
        temperature=temperature,
    )
    print("=== Modal multi-agg sectioning eval finished ===")
    print(f"run_id: {result['run_id']}")
    for label, m in result["results"].items():
        print(f"--- {label} ---")
        print("pair:", m["pair"])
        print("retrieval:", m["retrieval"])
    print(
        f"\nPull full metrics with:\n"
        f"  modal volume get {RESULTS_VOLUME} {result['run_id']} "
        f"./artifacts/voyage_nano_sectioned_modal/{result['run_id']}"
    )


@app.local_entrypoint()
def main(
    run_id: str = "",
    direction: str = "candidate",
    holdout_only: bool = False,
    user_id: str = "",
    model: str = "voyageai/voyage-4-nano",
    batch_size: int = 16,
    max_length: int = 8192,
    truncate_dim: int = 1024,
    gpu: str = "L4",
    agg: str = "max",
    topk: int = 2,
    temperature: float = 0.05,
) -> None:
    """CLI: modal run baselines/voyage_nano_sectioned/modal_eval.py --holdout-only"""
    if not run_id:
        run_id = f"holdout_{direction}" if holdout_only else f"user_{direction}"
        if direction == "seeker" and agg != "max":
            run_id += f"_{agg}"

    call = eval_remote
    if gpu and gpu != "L4":
        call = eval_remote.with_options(gpu=gpu)

    result = call.remote(
        run_id=run_id,
        direction=direction,
        holdout_only=holdout_only,
        user_id=user_id or None,
        model=model,
        batch_size=batch_size,
        max_length=max_length,
        truncate_dim=truncate_dim,
        agg=agg,
        topk=topk,
        temperature=temperature,
    )
    print("=== Modal sectioning eval finished ===")
    print(f"run_id: {result['run_id']}  direction: {result['direction']}  agg: {result.get('agg')}")
    print("pair:", result["pair"])
    print("retrieval:", result["retrieval"])
    print(
        f"\nPull full metrics with:\n"
        f"  modal volume get {RESULTS_VOLUME} {result['run_id']} "
        f"./artifacts/voyage_nano_sectioned_modal/{result['run_id']}"
    )
