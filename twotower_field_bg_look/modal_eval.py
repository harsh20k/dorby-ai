"""Modal GPU entrypoint: score the trimmed-both-sides adapter on all 200 real pairs.

Uses `twotower_field_bg_look.eval.run_eval` (this package's own trimmed-text
scorer), not `eval_real_full.eval.run_eval` — see `eval.py`'s module
docstring for why the shared path can't be reused unchanged here.

    modal run twotower_field_bg_look/modal_eval.py --run-id field_bg_look_001
    modal volume get dorby-twotower-field-bg-look-eval-results field_bg_look_001 \\
        ./artifacts/twotower_field_bg_look/field_bg_look_001_real200
"""

from __future__ import annotations

import modal

APP_NAME = "dorby-twotower-field-bg-look"
RESULTS_VOLUME = "dorby-twotower-field-bg-look-eval-results"
CHECKPOINT_VOLUME = "dorby-twotower-field-bg-look-checkpoints"
HF_CACHE_VOLUME = "dorby-twotower-hf-cache"

app = modal.App(APP_NAME)
GPU = "L4"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.51,<5",
        "sentence-transformers>=5.6,<6",
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
        "twotower_field_bg_look", "field_pairs_sweep", "eval_real_full", "twotower", "baselines", "synth_pipeline"
    )
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
    volumes={"/results": results, "/checkpoints": checkpoints, "/cache/huggingface": hf_cache},
)
def eval_remote(run_id: str, batch_size: int = 8) -> dict:
    from pathlib import Path

    from twotower_field_bg_look.eval import run_eval, write_metrics

    adapter_dir = Path("/checkpoints") / run_id / "adapter"
    metrics = run_eval(
        data_dir=Path("/root/data"),
        split_path=Path("/root/data/synthetic/seed_split.json"),
        model_name="voyageai/voyage-4-nano",
        adapter_dir=adapter_dir,
        batch_size=batch_size,
        device="cuda",
        truncate_dim=1024,
    )
    write_metrics(metrics, Path("/results") / run_id)
    results.commit()
    hf_cache.commit()

    out = {}
    for subset, m in metrics["subsets"].items():
        s = {
            "pair_auc": m["pair"]["roc_auc"],
            "hard_neg_auc": m["slices"]["neg_hardness"]["hard"]["pair_auc"],
            "mrr": m["retrieval"]["mrr"],
            "recall@1": m["retrieval"]["recall@1"],
            "recall@10": m["retrieval"]["recall@10"],
        }
        out[subset] = s
        print(
            f"{subset:8s} AUC={s['pair_auc']:.4f} hard={s['hard_neg_auc']:.4f} "
            f"MRR={s['mrr']:.4f} R@1={s['recall@1']:.4f} R@10={s['recall@10']:.4f}"
        )
    return out


@app.local_entrypoint()
def main(run_id: str = "field_bg_look_001", batch_size: int = 8) -> None:
    summary = eval_remote.remote(run_id, batch_size)
    print(f"\n=== {run_id}, all 200 real pairs (background+lookingFor, both sides) ===")
    for subset, s in summary.items():
        print(
            f"{subset:8s} AUC={s['pair_auc']:.4f} hard={s['hard_neg_auc']:.4f} "
            f"MRR={s['mrr']:.4f} R@1={s['recall@1']:.4f} R@10={s['recall@10']:.4f}"
        )
    print(
        f"\nPull with:\n  modal volume get {RESULTS_VOLUME} {run_id} "
        f"./artifacts/twotower_field_bg_look/{run_id}_real200"
    )
