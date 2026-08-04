"""Modal entrypoint: calibrate alpha on rrf_003, then report on all 200 real pairs.

One L4 session, one model load, two phases:

1. ``nomad_drift.calibrate.run_calibration`` on rrf_003 (2,619 synthetic pairs,
   923 unique profiles) picks the alpha that maximizes pair AUC.
2. That alpha is folded into ``query_weighted.eval.run_all_arms``'s own alpha
   grid and run unmodified on all 200 real pairs (subsets all/train/holdout) —
   the exact function that produced the published ``query_weighted`` table, so
   this run's numbers sit in the same table, not a reimplementation of it.
   ``nomad_drift.worked_example.pick_worked_example`` then picks one concrete
   before/after ranked list from the same encoded vectors, for the artifact.

    modal run nomad_drift/modal_run.py --run-id nd_001
    modal volume get dorby-nomad-drift-results nd_001 ./artifacts/nomad_drift/nd_001
"""

from __future__ import annotations

import modal

APP_NAME = "dorby-nomad-drift"
RESULTS_VOLUME = "dorby-nomad-drift-results"
EMB_CACHE_VOLUME = "dorby-nomad-drift-cache"
HF_CACHE_VOLUME = "dorby-twotower-hf-cache"  # shared model-download cache

app = modal.App(APP_NAME)

GPU = "L4"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.51,<5",
        "sentence-transformers>=3.4.1,<6",
        "accelerate>=0.30",
        "scikit-learn==1.9.0",
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
        "nomad_drift", "query_weighted", "eval_real_full", "baselines", "twotower", "synth_pipeline"
    )
    .add_local_dir("data", remote_path="/root/data")
    .add_local_dir("eval_real_full/data_frozen", remote_path="/root/eval_real_full/data_frozen")
    .add_local_file(
        "artifacts/pairing_rrf/rrf_003/manifest.json",
        remote_path="/root/artifacts/pairing_rrf/rrf_003/manifest.json",
    )
    .add_local_dir(
        "artifacts/pairing_rrf/rrf_003/staged",
        remote_path="/root/artifacts/pairing_rrf/rrf_003/staged",
    )
)

results = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True)
emb_cache = modal.Volume.from_name(EMB_CACHE_VOLUME, create_if_missing=True)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU,
    timeout=120 * 60,
    volumes={
        "/results": results,
        "/embcache": emb_cache,
        "/cache/huggingface": hf_cache,
    },
)
def run_remote(
    run_id: str = "nd_001",
    batch_size: int = 4,
    max_length: int = 4096,
    truncate_dim: int = 1024,
) -> dict:
    import json
    from pathlib import Path

    from baselines.voyage_nano.encode import VoyageNanoEncoder

    from nomad_drift.calibrate import RRF_BATCH_DEFAULT, run_calibration
    from nomad_drift.worked_example import pick_worked_example
    from query_weighted.eval import ALPHAS as QW_ALPHAS
    from query_weighted.eval import encode_everything, run_all_arms, write_metrics

    encoder = VoyageNanoEncoder(
        model_name="voyageai/voyage-4-nano",
        device="cuda",
        max_length=max_length,
        truncate_dim=truncate_dim,
        cache_dir=Path("/embcache"),
    )

    print("=== phase 1: calibrate alpha on rrf_003 ===")
    calibration = run_calibration(
        encoder, batch_dir=Path("/root") / RRF_BATCH_DEFAULT, batch_size=batch_size
    )
    best_alpha = calibration["best_alpha"]
    print(f"\ncalibrated alpha = {best_alpha}")

    print("\n=== phase 2: report on all 200 real pairs ===")
    report_alphas = tuple(sorted(set(QW_ALPHAS) | {best_alpha}))
    metrics = run_all_arms(
        encoder,
        Path("/root/data"),
        Path("/root/data/synthetic/seed_split.json"),
        subsets=("all", "train", "holdout"),
        batch_size=batch_size,
        alphas=report_alphas,
    )
    metrics["calibrated_alpha"] = best_alpha
    metrics["calibration"] = calibration

    print("\n=== phase 3: worked example (all-200 encoded vectors, reused) ===")
    enc = encode_everything(
        encoder,
        Path("/root/data"),
        Path("/root/data/synthetic/seed_split.json"),
        batch_size=batch_size,
        arms=["concat_baseline", "profile_only", "query_only"],
    )
    worked = pick_worked_example(enc, best_alpha)

    out_dir = Path("/results") / run_id
    write_metrics(metrics, out_dir)
    (out_dir / "worked_example.json").write_text(json.dumps(worked, indent=2) + "\n")
    (out_dir / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")

    results.commit()
    emb_cache.commit()
    hf_cache.commit()

    calibrated_key = f"alpha_{best_alpha:.1f}" if best_alpha not in (0.0, 1.0) else (
        "profile_only" if best_alpha == 0.0 else "query_only"
    )
    summary = {
        "run_id": run_id,
        "calibrated_alpha": best_alpha,
        "worked_example_pair_id": worked["pair_id"],
        "worked_example_improvement": worked["improvement"],
        "arms": {
            name: {
                "pair_auc": arm["all"]["pair"]["roc_auc"],
                "hard_neg_auc": arm["all"]["slices"]["neg_hardness"]["hard"]["pair_auc"],
                "mrr": arm["all"]["retrieval"]["mrr"],
                "recall@1": arm["all"]["retrieval"]["recall@1"],
                "recall@10": arm["all"]["retrieval"]["recall@10"],
            }
            for name, arm in metrics["arms"].items()
            if name in ("profile_only", "concat_baseline", "query_only", calibrated_key)
        },
    }
    return summary


@app.local_entrypoint()
def main(
    run_id: str = "nd_001",
    batch_size: int = 4,
    max_length: int = 4096,
    truncate_dim: int = 1024,
) -> None:
    res = run_remote.remote(
        run_id=run_id, batch_size=batch_size, max_length=max_length, truncate_dim=truncate_dim
    )
    print(f"\n=== nomad_drift ({res['run_id']}) ===")
    print(f"calibrated alpha (fit on rrf_003) = {res['calibrated_alpha']}")
    print(f"worked example pair: {res['worked_example_pair_id']} (rank improved by {res['worked_example_improvement']})")
    print(f"\n{'arm':18s} {'pairAUC':>8s} {'hardneg':>8s} {'MRR':>8s} {'R@1':>7s} {'R@10':>7s}")
    for name, s in res["arms"].items():
        print(
            f"{name:18s} {s['pair_auc']:8.4f} {s['hard_neg_auc']:8.4f} {s['mrr']:8.4f} "
            f"{s['recall@1']:7.4f} {s['recall@10']:7.4f}"
        )
    print(
        f"\nPull with:\n  modal volume get {RESULTS_VOLUME} {run_id} "
        f"./artifacts/nomad_drift/{run_id}"
    )
