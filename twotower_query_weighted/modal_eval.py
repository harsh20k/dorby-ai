"""Modal GPU entrypoint: alpha-sweep the fine-tuned two-tower adapter on all 200 real pairs.

Inference only (no training) — mirrors the shape of ``eval_real_full/modal_eval.py``
but runs the query-weighting sweep from ``twotower_query_weighted/eval.py`` instead
of a single concat-baseline pass. Own app and results volume; shares only the
read/write HF model cache and the adapter checkpoints already on disk.

    modal run twotower_query_weighted/modal_eval.py --run-id qw_top1_ctrl_001

    modal volume get dorby-twotower-query-weighted-results <run_id> \\
        ./artifacts/twotower_query_weighted/<run_id>
"""

from __future__ import annotations

import modal

APP_NAME = "dorby-twotower-query-weighted"
RESULTS_VOLUME = "dorby-twotower-query-weighted-results"
HF_CACHE_VOLUME = "dorby-twotower-hf-cache"  # shared read/write cache

app = modal.App(APP_NAME)

GPU = "L4"  # inference only, voyage-4-nano at 4096 tokens — same as eval_real_full

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.51,<5",
        "sentence-transformers>=3.4.1,<6",
        "peft>=0.14,<1",
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
    .add_local_python_source(
        "twotower_query_weighted", "query_weighted", "eval_real_full", "twotower", "baselines", "synth_pipeline"
    )
    .add_local_dir("eval_real_full/data_frozen", remote_path="/root/eval_real_full/data_frozen")
    .add_local_dir("data", remote_path="/root/data")
    .add_local_dir("artifacts/twotower_top1_optimised", remote_path="/root/top1_runs")
)

results = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True)

# label -> adapter to sweep. Currently just the project's best fine-tune.
CONFIGS: dict[str, dict] = {
    "top1_ctrl": {
        "adapter": "/root/top1_runs/top1_ctrl_001/adapter",
        "model": "voyageai/voyage-4-nano",
    },
}


@app.function(
    image=image,
    gpu=GPU,
    timeout=90 * 60,
    volumes={"/results": results, "/cache/huggingface": hf_cache},
)
def eval_remote(run_id: str, config: str, batch_size: int = 8) -> dict:
    from pathlib import Path

    from twotower_query_weighted.eval import load_adapter_model, run_all_arms, write_metrics

    spec = CONFIGS[config]
    model = load_adapter_model(
        model_name=spec["model"],
        adapter_dir=Path(spec["adapter"]),
        device="cuda",
        max_length=4096,
        truncate_dim=1024,
    )
    metrics = run_all_arms(
        model,
        data_dir=Path("/root/data"),
        split_path=Path("/root/data/synthetic/seed_split.json"),
        label=config,
        batch_size=batch_size,
    )
    write_metrics(metrics, Path("/results") / run_id / config)
    results.commit()
    hf_cache.commit()
    return {
        "config": config,
        "summary": {
            arm: {
                "pair_auc": subsets["all"]["pair"]["roc_auc"],
                "mrr": subsets["all"]["retrieval"]["mrr"],
                "recall@1": subsets["all"]["retrieval"]["recall@1"],
                "recall@10": subsets["all"]["retrieval"]["recall@10"],
            }
            for arm, subsets in metrics["arms"].items()
        },
    }


@app.local_entrypoint()
def main(run_id: str = "qw_top1_ctrl_001", configs: str = "top1_ctrl", batch_size: int = 8) -> None:
    wanted = [c.strip() for c in configs.split(",") if c.strip()]
    unknown = [c for c in wanted if c not in CONFIGS]
    if unknown:
        raise SystemExit(f"unknown configs {unknown}; choices: {list(CONFIGS)}")

    print(f"run_id={run_id} configs={wanted}")
    handles = [eval_remote.spawn(run_id, c, batch_size) for c in wanted]
    for h in handles:
        res = h.get()
        print(f"\n=== {res['config']} ===")
        for arm, s in sorted(res["summary"].items()):
            print(
                f"  {arm:16s} AUC={s['pair_auc']:.4f} MRR={s['mrr']:.4f} "
                f"R@1={s['recall@1']:.4f} R@10={s['recall@10']:.4f}"
            )
    print(
        f"\nPull with:\n  modal volume get {RESULTS_VOLUME} {run_id} "
        f"./artifacts/twotower_query_weighted/{run_id}"
    )
