"""Modal entrypoint: score a trained ask/offer adapter on holdout + all-200,
reading the adapter straight from the training run's checkpoint volume (no
local download needed first).

  modal run twotower_ask_offer/modal_eval.py --run-id ask_offer_001
"""

from __future__ import annotations

import modal

from twotower_ask_offer.modal_train import CHECKPOINT_VOLUME, HF_CACHE_VOLUME, GPU, image

# Own App — importing modal_train's `app` object also re-registers its
# `main` local entrypoint, which collides with this module's own `main`
# (Modal requires local-entrypoint names to be unique per App).
app = modal.App("dorby-twotower-ask-offer-eval")

checkpoints = modal.Volume.from_name(CHECKPOINT_VOLUME, create_if_missing=False)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU,
    timeout=30 * 60,
    volumes={"/checkpoints": checkpoints, "/cache/huggingface": hf_cache},
)
def eval_remote(run_id: str, lam: float) -> dict:
    from pathlib import Path

    from twotower_ask_offer.config import TOWER_CONFIG
    from twotower_ask_offer.eval import run_eval

    adapter_dir = Path("/checkpoints") / run_id / "adapter"
    metrics = run_eval(
        Path("/root/data"), Path("/root/data/synthetic/seed_split.json"), adapter_dir,
        lam=lam, tower_cfg=TOWER_CONFIG, batch_size=8,
    )
    out_path = Path("/checkpoints") / run_id / "metrics_full_eval.json"
    import json

    out_path.write_text(json.dumps(metrics, indent=2) + "\n")
    checkpoints.commit()
    return metrics


@app.local_entrypoint()
def main(run_id: str = "ask_offer_001", lam: float = 1.75) -> None:
    import json

    metrics = eval_remote.remote(run_id=run_id, lam=lam)
    print(json.dumps(metrics, indent=2))
