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
# 8B in bf16 is ~16GB resident before activations; L4's 24GB is too tight.
GPU_LARGE = "A100-80GB"

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
    .add_local_dir(
        "artifacts/twotower_qwen_bigbatch", remote_path="/root/qwen_runs"
    )
    .add_local_dir(
        "artifacts/twotower_top1_optimised", remote_path="/root/top1_runs"
    )
    .add_local_dir(
        "artifacts/twotower_voyage_gemini_ctrl", remote_path="/root/voyage_gemini_ctrl_runs"
    )
    .add_local_dir(
        "artifacts/twotower_voyage_gemini_kl", remote_path="/root/voyage_gemini_kl_runs"
    )
    .add_local_dir(
        "artifacts/twotower_qwen_voyage_gemini", remote_path="/root/qwen_voyage_gemini_runs"
    )
)

results = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True)

# label -> what to evaluate. `adapter` is None for a frozen base model.
# `dtype` None keeps the shared fp32 load path that every prior published number
# used; bfloat16 is only for backbones too large to load fp32, and is never
# mixed inside one model-family comparison.
CONFIGS: dict[str, dict] = {
    # voyage-4-nano family — small, fp32, L4 is plenty
    "frozen": {"adapter": None, "model": "voyageai/voyage-4-nano", "dtype": None, "size": "small"},
    "arm_a_v1": {
        "adapter": "/root/ablation_runs/abl_a_batch_only/adapter",
        "model": "voyageai/voyage-4-nano",
        "dtype": None,
        "size": "small",
    },
    "arm_a_v2": {
        "adapter": "/root/ablation_runs/abl_a_batch_only_v2/adapter",
        "model": "voyageai/voyage-4-nano",
        "dtype": None,
        "size": "small",
    },
    # Qwen3-Embedding-8B family — bf16 throughout, including the frozen control,
    # so the fine-tune is never compared against a different-precision baseline.
    "qwen_frozen": {
        "adapter": None,
        "model": "Qwen/Qwen3-Embedding-8B",
        "dtype": "bfloat16",
        "size": "large",
    },
    # Frozen control at Qwen's native 4096 dims. The fine-tunes trained with
    # truncate_dim=1024 and must be scored there, but that truncation is itself
    # a handicap the frozen model never had in its published 0.6595 baseline
    # (which used truncate_dim=None). Without this arm it is impossible to tell
    # whether frozen Qwen's weak all-200 showing is the model or the truncation.
    "qwen_frozen_4096": {
        "adapter": None,
        "model": "Qwen/Qwen3-Embedding-8B",
        "dtype": "bfloat16",
        "size": "large",
        "truncate_dim": None,
    },
    "qwen_micro1": {
        "adapter": "/root/qwen_runs/qwen_micro1_r1/adapter",
        "model": "Qwen/Qwen3-Embedding-8B",
        "dtype": "bfloat16",
        "size": "large",
    },
    "qwen_micro6": {
        "adapter": "/root/qwen_runs/qwen_micro6_r1/adapter",
        "model": "Qwen/Qwen3-Embedding-8B",
        "dtype": "bfloat16",
        "size": "large",
    },
    # Qwen micro-6 recipe retrained on the pairing_voyage_gemini batch instead
    # of rrf_003 (see twotower_qwen_voyage_gemini/)
    "qwen_voyage_gemini": {
        "adapter": "/root/qwen_voyage_gemini_runs/qwen_voyage_gemini_001_b200_v2/adapter",
        "model": "Qwen/Qwen3-Embedding-8B",
        "dtype": "bfloat16",
        "size": "large",
    },
    # recall@1-targeted arm and its control (see twotower_top1_optimised/)
    "top1_ctrl": {
        "adapter": "/root/top1_runs/top1_ctrl_001/adapter",
        "model": "voyageai/voyage-4-nano",
        "dtype": None,
        "size": "small",
    },
    "top1_sharp": {
        "adapter": "/root/top1_runs/top1_001/adapter",
        "model": "voyageai/voyage-4-nano",
        "dtype": None,
        "size": "small",
    },
    # top1_ctrl's exact recipe, retrained on the pairing_voyage_gemini batch
    # instead of rrf_003 (see twotower_voyage_gemini_ctrl/)
    "voyage_gemini_ctrl": {
        "adapter": "/root/voyage_gemini_ctrl_runs/voyage_gemini_ctrl_001/adapter",
        "model": "voyageai/voyage-4-nano",
        "dtype": None,
        "size": "small",
    },
    # voyage_gemini_ctrl_001's recipe/data + a KL leash to the frozen base model
    # (see twotower_voyage_gemini_kl/)
    "voyage_gemini_kl": {
        "adapter": "/root/voyage_gemini_kl_runs/voyage_gemini_kl_001/adapter",
        "model": "voyageai/voyage-4-nano",
        "dtype": None,
        "size": "small",
    },
}


def _run_one(run_id: str, config: str, batch_size: int) -> dict:
    """Shared body — identical for both GPU classes, so only the GPU differs."""
    from pathlib import Path

    from eval_real_full.eval import run_eval, write_metrics

    spec = CONFIGS[config]
    adapter_dir = Path(spec["adapter"]) if spec["adapter"] else None

    metrics = run_eval(
        data_dir=Path("/root/data"),
        split_path=Path("/root/data/synthetic/seed_split.json"),
        model_name=spec["model"],
        adapter_dir=adapter_dir,
        label=config,
        batch_size=batch_size,
        device="cuda",
        torch_dtype=spec["dtype"],
        # Default 1024 matches what the fine-tunes trained with; a config may
        # override it (e.g. the native-dimension frozen control).
        truncate_dim=spec.get("truncate_dim", 1024),
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


@app.function(
    image=image,
    gpu=GPU,
    timeout=90 * 60,
    volumes={"/results": results, "/cache/huggingface": hf_cache},
)
def eval_remote(run_id: str, config: str, batch_size: int = 8) -> dict:
    return _run_one(run_id, config, batch_size)


@app.function(
    image=image,
    gpu=GPU_LARGE,
    timeout=180 * 60,
    volumes={"/results": results, "/cache/huggingface": hf_cache},
)
def eval_remote_large(run_id: str, config: str, batch_size: int = 4) -> dict:
    """Same body, bigger card — an 8B backbone in bf16 needs ~16GB resident."""
    return _run_one(run_id, config, batch_size)


@app.local_entrypoint()
def main(run_id: str = "real200_001", configs: str = "", batch_size: int = 8) -> None:
    """Run each config in parallel; ``--configs`` is a comma-separated subset."""
    wanted = [c.strip() for c in configs.split(",") if c.strip()] or list(CONFIGS)
    unknown = [c for c in wanted if c not in CONFIGS]
    if unknown:
        raise SystemExit(f"unknown configs {unknown}; choices: {list(CONFIGS)}")

    print(f"run_id={run_id} configs={wanted}")
    handles = [
        (
            eval_remote_large.spawn(run_id, c, max(1, batch_size // 2))
            if CONFIGS[c]["size"] == "large"
            else eval_remote.spawn(run_id, c, batch_size)
        )
        for c in wanted
    ]
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
