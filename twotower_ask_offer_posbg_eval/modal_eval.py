"""Modal GPU: rescore ask_offer_001 adapters on all 200 real pairs.

Offer text = positioning + background only. No holdout split.

  modal run twotower_ask_offer_posbg_eval/modal_eval.py
"""

from __future__ import annotations

import modal

APP_NAME = "dorby-twotower-ask-offer-posbg-eval"
CHECKPOINT_VOLUME = "dorby-twotower-ask-offer-checkpoints"
HF_CACHE_VOLUME = "dorby-twotower-hf-cache"
RESULTS_VOLUME = "dorby-twotower-ask-offer-posbg-eval"

app = modal.App(APP_NAME)

GPU = "A100-80GB"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.51,<5",
        "sentence-transformers>=5.6,<6",
        "peft>=0.14,<1",
        "accelerate>=0.30",
        "datasets>=2.19",
        "scikit-learn>=1.4.0",
        "numpy>=1.26.0",
        "tqdm>=4.66.0",
        "python-dotenv>=1.0.0",
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
        "twotower_ask_offer_posbg_eval",
        "twotower_ask_offer",
        "twotower",
        "baselines",
        "eval_real_full",
        "synth_pipeline",
    )
    .add_local_dir("data", remote_path="/root/data")
    .add_local_dir("eval_real_full/data_frozen", remote_path="/root/eval_real_full/data_frozen")
)

checkpoints = modal.Volume.from_name(CHECKPOINT_VOLUME, create_if_missing=False)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True)
results = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU,
    timeout=45 * 60,
    volumes={
        "/checkpoints": checkpoints,
        "/cache/huggingface": hf_cache,
        "/results": results,
    },
)
def eval_remote(run_id: str, lam: float) -> dict:
    import json
    from pathlib import Path

    from twotower_ask_offer.config import TOWER_CONFIG
    from twotower_ask_offer_posbg_eval.eval import print_summary, run_eval

    adapter_dir = Path("/checkpoints") / run_id / "adapter"
    metrics = run_eval(
        Path("/root/data"),
        Path("/root/data/synthetic/seed_split.json"),
        adapter_dir,
        lam=lam,
        tower_cfg=TOWER_CONFIG,
        batch_size=8,
    )
    print_summary(metrics)
    out_dir = Path("/results") / "all200"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    results.commit()
    return metrics


@app.local_entrypoint()
def main(run_id: str = "ask_offer_001", lam: float = 1.75) -> None:
    import json

    metrics = eval_remote.remote(run_id=run_id, lam=lam)
    print(json.dumps(metrics, indent=2))
    print(
        "\nPull metrics with:\n"
        f"  modal volume get {RESULTS_VOLUME} all200 "
        "./artifacts/twotower_ask_offer_posbg_eval"
    )
