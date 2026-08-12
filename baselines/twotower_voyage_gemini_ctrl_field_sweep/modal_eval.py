"""Modal GPU entrypoint for the 105-combo voyage_gemini_ctrl field/query sweep.

One remote call loads voyage-4-nano + the voyage_gemini_ctrl LoRA adapter
once and runs all 105 combos in-process (sweep.run_sweep dedups encoding to
22 unique encode groups instead of 105 — see sweep.py's docstring), so this
needs one GPU boot, not 105.

Usage:
  modal run baselines/twotower_voyage_gemini_ctrl_field_sweep/modal_eval.py
  modal volume get dorby-twotower-voyage-gemini-ctrl-field-sweep-eval real_all ./artifacts/twotower_voyage_gemini_ctrl_field_sweep_modal/real_all
"""

from __future__ import annotations

import modal

APP_NAME = "dorby-twotower-voyage-gemini-ctrl-field-sweep"
RESULTS_VOLUME = "dorby-twotower-voyage-gemini-ctrl-field-sweep-eval"
HF_CACHE_VOLUME = "dorby-twotower-hf-cache"  # reuse the existing HF model cache
ADAPTER_LOCAL_DIR = "artifacts/twotower_voyage_gemini_ctrl/voyage_gemini_ctrl_001/adapter"
ADAPTER_REMOTE_DIR = "/root/adapter"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.51,<5",
        "sentence-transformers>=3.4.1,<6",
        "peft>=0.11.0,<1",
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
    .add_local_python_source("baselines", "synth_pipeline", "twotower")
    .add_local_dir("data", remote_path="/root/data")
    .add_local_dir(ADAPTER_LOCAL_DIR, remote_path=ADAPTER_REMOTE_DIR)
)

results = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=90 * 60,
    volumes={
        "/results": results,
        "/cache/huggingface": hf_cache,
    },
)
def sweep_remote(
    run_id: str = "real_all",
    model: str = "voyageai/voyage-4-nano",
    batch_size: int = 16,
    max_length: int = 4096,
    truncate_dim: int = 1024,
) -> dict:
    import json
    from pathlib import Path

    from baselines.twotower_voyage_gemini_ctrl_field_sweep.sweep import run_sweep

    artifacts_dir = Path("/results") / run_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    sweep_results = run_sweep(
        data_dir=Path("/root/data"),
        adapter_dir=Path(ADAPTER_REMOTE_DIR),
        model_name=model,
        batch_size=batch_size,
        max_length=max_length,
        truncate_dim=truncate_dim,
        artifacts_dir=artifacts_dir,
        split_path=Path("/root/data/synthetic/seed_split.json"),
    )
    out_path = artifacts_dir / "sweep_results.json"
    out_path.write_text(json.dumps(sweep_results, indent=2) + "\n")
    results.commit()

    ranked = sorted(sweep_results, key=lambda r: -r["pair"]["roc_auc"])
    return {
        "run_id": run_id,
        "num_combos": len(sweep_results),
        "top5": [
            {"combo_id": r["combo_id"], "pair_auc": r["pair"]["roc_auc"]} for r in ranked[:5]
        ],
    }


@app.local_entrypoint()
def main(
    run_id: str = "real_all",
    model: str = "voyageai/voyage-4-nano",
    batch_size: int = 16,
    max_length: int = 4096,
    truncate_dim: int = 1024,
    gpu: str = "A100-40GB",
) -> None:
    """CLI: modal run baselines/twotower_voyage_gemini_ctrl_field_sweep/modal_eval.py"""
    call = sweep_remote
    if gpu and gpu != "A100-40GB":
        call = sweep_remote.with_options(gpu=gpu)

    result = call.remote(
        run_id=run_id,
        model=model,
        batch_size=batch_size,
        max_length=max_length,
        truncate_dim=truncate_dim,
    )
    print("=== Modal voyage_gemini_ctrl field/query sweep finished ===")
    print(f"run_id: {result['run_id']}")
    print(f"combos: {result['num_combos']}")
    print("top 5 by pair AUC:")
    for row in result["top5"]:
        print(f"  {row['pair_auc']:.4f}  {row['combo_id']}")
    print(
        f"\nPull full results with:\n"
        f"  modal volume get {RESULTS_VOLUME} {result['run_id']} "
        f"./artifacts/twotower_voyage_gemini_ctrl_field_sweep_modal/{result['run_id']}"
    )
