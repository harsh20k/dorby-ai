"""Modal GPU entrypoint: holdout (69-pair) sanity-check eval for the
voyage_gemini_kl adapter.

Runs `twotower_voyage_gemini_kl.eval.run_holdout_eval` on Modal instead of
locally — model loading and embedding computation must happen on the GPU
container, not on this machine, matching the project's "compute on Modal,
pull only small result files" pattern already used for training. Only the
resulting metrics dict comes back to the local machine; no raw embeddings
are pulled.

Deliberately narrow in scope, same as `eval.py`: holdout-only, not the
all-200 number this project's standing rule says decides anything. See
`eval.py`'s module docstring and this package's `__init__.py` for why
all-200 scoring is out of scope here.

    modal run twotower_voyage_gemini_kl/modal_eval.py --run-id voyage_gemini_kl_001
"""

from __future__ import annotations

import modal

APP_NAME = "dorby-twotower-voyage-gemini-kl"
CHECKPOINT_VOLUME = "dorby-twotower-voyage-gemini-kl-checkpoints"
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
    .add_local_python_source("twotower_voyage_gemini_kl", "twotower", "baselines", "synth_pipeline")
    .add_local_dir("data", remote_path="/root/data")
)

checkpoints = modal.Volume.from_name(CHECKPOINT_VOLUME, create_if_missing=True)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU,
    timeout=30 * 60,
    volumes={"/checkpoints": checkpoints, "/cache/huggingface": hf_cache},
)
def eval_remote(run_id: str, batch_size: int = 6) -> dict:
    from pathlib import Path

    from twotower_voyage_gemini_kl.eval import run_holdout_eval

    adapter_dir = Path("/checkpoints") / run_id / "adapter"
    metrics = run_holdout_eval(
        adapter_dir=adapter_dir,
        data_dir=Path("/root/data"),
        split_path=Path("/root/data/synthetic/seed_split.json"),
        model_name="voyageai/voyage-4-nano",
        batch_size=batch_size,
        max_length=4096,
        truncate_dim=1024,
    )
    hf_cache.commit()
    return {
        "pair_auc": metrics["pair"]["roc_auc"],
        "pair_ap": metrics["pair"]["average_precision"],
        "hard_negative_auc": metrics["slices"]["neg_hardness"]["hard"]["pair_auc"],
        "easy_negative_auc": metrics["slices"]["neg_hardness"]["easy"]["pair_auc"],
        "mrr": metrics["retrieval"]["mrr"],
        "recall@1": metrics["retrieval"]["recall@1"],
        "recall@10": metrics["retrieval"]["recall@10"],
        "n_pos": metrics["n_pos"],
        "n_neg": metrics["n_neg"],
        "full_metrics": metrics,
    }


@app.local_entrypoint()
def main(run_id: str = "voyage_gemini_kl_001", batch_size: int = 6) -> None:
    import json
    from pathlib import Path

    summary = eval_remote.remote(run_id, batch_size)
    print(f"\n=== {run_id}, holdout sanity check (69 real pairs) ===")
    for k in ("pair_auc", "pair_ap", "hard_negative_auc", "easy_negative_auc", "mrr", "recall@1", "recall@10"):
        print(f"{k}: {summary[k]}")

    out_dir = Path("artifacts/twotower_voyage_gemini_kl") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "metrics_holdout_sanity.json"
    out.write_text(json.dumps(summary["full_metrics"], indent=2) + "\n")
    print(f"\nwrote {out} (locally, from the small returned metrics dict only)")
