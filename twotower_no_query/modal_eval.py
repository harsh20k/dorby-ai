"""Modal GPU entrypoint: score no_query_001 on all 200 real pairs.

Inference only — reuses `eval_real_full.eval.run_eval` unchanged (the same
path every other adapter in this project, including top1_ctrl, was scored
through), so the numbers are directly comparable. Own app/volume; shares only
the HF model-download cache.

    modal run twotower_no_query/modal_eval.py --run-id no_query_001

    modal volume get dorby-twotower-no-query-eval-results no_query_001 \\
        ./artifacts/twotower_no_query/no_query_001_real200
"""

from __future__ import annotations

import modal

APP_NAME = "dorby-twotower-no-query"
RESULTS_VOLUME = "dorby-twotower-no-query-eval-results"
CHECKPOINT_VOLUME = "dorby-twotower-no-query-checkpoints"  # read-only reuse
HF_CACHE_VOLUME = "dorby-twotower-hf-cache"

app = modal.App(APP_NAME)

GPU = "L4"  # inference only, voyage-4-nano — same class eval_real_full uses

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
    .add_local_python_source("twotower_no_query", "eval_real_full", "twotower", "baselines", "synth_pipeline")
    .add_local_dir("eval_real_full/data_frozen", remote_path="/root/eval_real_full/data_frozen")
    .add_local_dir("data", remote_path="/root/data")
)

results = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True)
checkpoints = modal.Volume.from_name(CHECKPOINT_VOLUME, create_if_missing=True)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU,
    timeout=60 * 60,
    volumes={
        "/results": results,
        "/checkpoints": checkpoints,
        "/cache/huggingface": hf_cache,
    },
)
def eval_remote(run_id: str, batch_size: int = 8) -> dict:
    from pathlib import Path

    from eval_real_full.eval import run_eval, write_metrics

    metrics = run_eval(
        data_dir=Path("/root/data"),
        split_path=Path("/root/data/synthetic/seed_split.json"),
        model_name="voyageai/voyage-4-nano",
        adapter_dir=Path("/checkpoints") / run_id / "adapter",
        label="no_query",
        batch_size=batch_size,
        device="cuda",
        truncate_dim=1024,
    )
    write_metrics(metrics, Path("/results") / run_id)
    results.commit()
    hf_cache.commit()
    return {
        subset: {
            "pair_auc": m["pair"]["roc_auc"],
            "hard_neg_auc": m["slices"]["neg_hardness"]["hard"]["pair_auc"],
            "mrr": m["retrieval"]["mrr"],
            "recall@1": m["retrieval"]["recall@1"],
            "recall@10": m["retrieval"]["recall@10"],
            "n_candidates": m["n_candidates"],
        }
        for subset, m in metrics["subsets"].items()
    }


@app.local_entrypoint()
def main(run_id: str = "no_query_001", batch_size: int = 8) -> None:
    summary = eval_remote.remote(run_id, batch_size)
    print(f"\n=== no_query eval, all 200 real pairs ({run_id}) ===")
    for subset, s in summary.items():
        print(
            f"  {subset:8s} corpus={s['n_candidates']:4d} "
            f"AUC={s['pair_auc']:.4f} hard={s['hard_neg_auc']:.4f} "
            f"MRR={s['mrr']:.4f} R@1={s['recall@1']:.4f} R@10={s['recall@10']:.4f}"
        )
    print(
        f"\nPull with:\n  modal volume get {RESULTS_VOLUME} {run_id} "
        f"./artifacts/twotower_no_query/{run_id}_real200"
    )
