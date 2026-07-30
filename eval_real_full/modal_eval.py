"""Modal GPU entrypoint: frozen nano vs. Arm A adapters on all 200 real pairs.

Own app and own results volume; shares only the read/write HF model cache with
the training packages. No training happens here — this is inference only, so it
runs on a much smaller GPU than the ablation needed.

    modal run eval_real_full/modal_eval.py                      # all 3 configs
    modal run eval_real_full/modal_eval.py --configs frozen      # just one

    modal volume get dorby-eval-real-full-results <run_id> \\
        ./artifacts/eval_real_full/<run_id>
"""

from __future__ import annotations

import modal

APP_NAME = "dorby-eval-real-full"
RESULTS_VOLUME = "dorby-eval-real-full-results"
HF_CACHE_VOLUME = "dorby-twotower-hf-cache"  # shared read/write cache

app = modal.App(APP_NAME)

# Inference only — no optimizer state, no activations retained. L4 is plenty for
# voyage-4-nano at 4096 tokens and far cheaper than the A100 training needed.
GPU = "L4"

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
    .add_local_python_source("eval_real_full", "twotower", "baselines", "synth_pipeline")
    # add_local_python_source ships only .py files, so the frozen provenance
    # manifest (JSON) has to be mounted explicitly or data.py's verify fails.
    .add_local_dir(
        "eval_real_full/data_frozen", remote_path="/root/eval_real_full/data_frozen"
    )
    .add_local_dir("data", remote_path="/root/data")
    .add_local_dir(
        "artifacts/twotower_rrf_triplet_ablation", remote_path="/root/ablation_runs"
    )
)

results = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True)

# label -> adapter directory under /root/ablation_runs (None = frozen base model)
CONFIGS: dict[str, str | None] = {
    "frozen": None,
    "arm_a_v1": "abl_a_batch_only/adapter",
    "arm_a_v2": "abl_a_batch_only_v2/adapter",
}


@app.function(
    image=image,
    gpu=GPU,
    timeout=90 * 60,
    volumes={"/results": results, "/cache/huggingface": hf_cache},
)
def eval_remote(run_id: str, config: str, batch_size: int = 8) -> dict:
    from pathlib import Path

    from eval_real_full.eval import run_eval, write_metrics

    rel = CONFIGS[config]
    adapter_dir = Path("/root/ablation_runs") / rel if rel else None

    metrics = run_eval(
        data_dir=Path("/root/data"),
        split_path=Path("/root/data/synthetic/seed_split.json"),
        adapter_dir=adapter_dir,
        label=config,
        batch_size=batch_size,
        device="cuda",
    )
    write_metrics(metrics, Path("/results") / run_id / config)
    results.commit()
    hf_cache.commit()
    return {
        "config": config,
        "summary": {
            subset: {
                "pair_auc": m["pair"]["roc_auc"],
                "hard_neg_auc": m["slices"]["neg_hardness"]["hard"]["pair_auc"],
                "mrr": m["retrieval"]["mrr"],
                "recall@1": m["retrieval"]["recall@1"],
                "recall@10": m["retrieval"]["recall@10"],
                "n_candidates": m["n_candidates"],
            }
            for subset, m in metrics["subsets"].items()
        },
    }


@app.local_entrypoint()
def main(run_id: str = "real200_001", configs: str = "", batch_size: int = 8) -> None:
    """Run each config in parallel; ``--configs`` is a comma-separated subset."""
    wanted = [c.strip() for c in configs.split(",") if c.strip()] or list(CONFIGS)
    unknown = [c for c in wanted if c not in CONFIGS]
    if unknown:
        raise SystemExit(f"unknown configs {unknown}; choices: {list(CONFIGS)}")

    print(f"run_id={run_id} configs={wanted}")
    handles = [eval_remote.spawn(run_id, c, batch_size) for c in wanted]
    for h in handles:
        res = h.get()
        print(f"\n=== {res['config']} ===")
        for subset, s in res["summary"].items():
            print(
                f"  {subset:8s} corpus={s['n_candidates']:4d} "
                f"AUC={s['pair_auc']:.4f} hard={s['hard_neg_auc']:.4f} "
                f"MRR={s['mrr']:.4f} R@1={s['recall@1']:.4f} R@10={s['recall@10']:.4f}"
            )
    print(
        f"\nPull with:\n  modal volume get {RESULTS_VOLUME} {run_id} "
        f"./artifacts/eval_real_full/{run_id}"
    )
