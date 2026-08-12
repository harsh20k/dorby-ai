"""Modal entrypoint: lambda sweep on ask_offer_001, reading its adapters
straight from twotower_ask_offer's checkpoint volume (no local download
needed first). Own App/image — mirrors twotower_ask_offer/modal_eval.py's
volume-reuse pattern, substituting the sweep eval for the fixed-lambda one.

  modal run reciprocal_lambda_grid_ask_offer/modal_eval.py --run-id ask_offer_001
  modal volume get dorby-reciprocal-lambda-grid-ask-offer-eval ask_offer_001 \\
      ./artifacts/reciprocal_lambda_grid_ask_offer
"""

from __future__ import annotations

import modal

from twotower_ask_offer.modal_train import CHECKPOINT_VOLUME, GPU, HF_CACHE_VOLUME
from twotower_ask_offer.modal_train import image as base_image

app = modal.App("dorby-reciprocal-lambda-grid-ask-offer-eval")

image = base_image.add_local_python_source("reciprocal_lambda_grid_ask_offer")

RESULTS_VOLUME = "dorby-reciprocal-lambda-grid-ask-offer-eval"

checkpoints = modal.Volume.from_name(CHECKPOINT_VOLUME, create_if_missing=False)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True)
results = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU,
    timeout=30 * 60,
    volumes={"/checkpoints": checkpoints, "/cache/huggingface": hf_cache, "/results": results},
)
def eval_remote(
    run_id: str,
    train_run_id: str = "ask_offer_001",
    batch_size: int = 8,
    lambda_min: float = -2.0,
    lambda_max: float = 2.0,
    lambda_step: float = 0.05,
) -> dict:
    import json
    from pathlib import Path

    from baselines.reciprocal_static.eval import build_lambda_grid
    from reciprocal_lambda_grid_ask_offer.eval import run_eval

    adapter_dir = Path("/checkpoints") / train_run_id / "adapter"
    artifacts_dir = Path("/results") / run_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    lambda_grid = build_lambda_grid(lambda_min, lambda_max, lambda_step)
    metrics = run_eval(
        data_dir=Path("/root/data"),
        split_path=Path("/root/data/synthetic/seed_split.json"),
        adapter_dir=adapter_dir,
        batch_size=batch_size,
        lambda_grid=lambda_grid,
    )
    (artifacts_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    results.commit()
    return metrics


@app.local_entrypoint()
def main(
    run_id: str = "ask_offer_001",
    train_run_id: str = "ask_offer_001",
    batch_size: int = 8,
    lambda_min: float = -2.0,
    lambda_max: float = 2.0,
    lambda_step: float = 0.05,
) -> None:
    metrics = eval_remote.remote(
        run_id=run_id,
        train_run_id=train_run_id,
        batch_size=batch_size,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        lambda_step=lambda_step,
    )
    print("=== Modal reciprocal-lambda-grid-ask-offer eval finished ===")
    for name in ("holdout", "all"):
        s = metrics["subsets"][name]
        print(
            f"{name}: forward-only AUC={s['forward_only_auc']:.4f} "
            f"curve max=(lambda={s['best_lambda']:.2f}, AUC={s['best_auc']:.4f})"
        )
    print(
        f"\nPull full metrics with:\n  modal volume get {RESULTS_VOLUME} {run_id} "
        f"./artifacts/reciprocal_lambda_grid_ask_offer_{run_id}"
    )
