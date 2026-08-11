"""Modal entrypoint: calibrate query+section alpha on rrf_003, confirm on all 200 real pairs.

    modal run nomad_drift_sectioned/modal_run.py --run-id nds_001
    modal volume get dorby-nomad-drift-sectioned-results nds_001/metrics.json \\
        ./artifacts/nomad_drift_sectioned/nds_001/metrics.json
    modal volume get dorby-nomad-drift-sectioned-results nds_001/calibration.json \\
        ./artifacts/nomad_drift_sectioned/nds_001/calibration.json
"""

from __future__ import annotations

import modal

APP_NAME = "dorby-nomad-drift-sectioned"
RESULTS_VOLUME = "dorby-nomad-drift-sectioned-results"
EMB_CACHE_VOLUME = "dorby-nomad-drift-sectioned-cache"
HF_CACHE_VOLUME = "dorby-twotower-hf-cache"

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
        "nomad_drift_sectioned",
        "nomad_drift",
        "query_weighted",
        "eval_real_full",
        "baselines",
        "twotower",
        "synth_pipeline",
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
    run_id: str = "nds_001",
    batch_size: int = 4,
    max_length: int = 4096,
    truncate_dim: int = 1024,
) -> dict:
    import json
    from pathlib import Path

    from baselines.voyage_nano.encode import VoyageNanoEncoder
    from nomad_drift.calibrate import RRF_BATCH_DEFAULT, ALPHAS as ND_ALPHAS
    from nomad_drift_sectioned.calibrate import run_sectioned_calibration
    from nomad_drift_sectioned.report import build_section_report_encoding, score_alphas

    encoder = VoyageNanoEncoder(
        model_name="voyageai/voyage-4-nano",
        device="cuda",
        max_length=max_length,
        truncate_dim=truncate_dim,
        cache_dir=Path("/embcache"),
    )

    print("=== phase 1: calibrate query+section alpha on rrf_003 ===")
    calibration = run_sectioned_calibration(
        encoder, batch_dir=Path("/root") / RRF_BATCH_DEFAULT, batch_size=batch_size
    )
    best_alpha = calibration["best_alpha"]
    print(f"\ncalibrated alpha = {best_alpha}")

    print("\n=== phase 2: report on all 200 real pairs (full alpha grid) ===")
    sre = build_section_report_encoding(
        encoder, Path("/root/data"), Path("/root/data/synthetic/seed_split.json"), batch_size=batch_size
    )
    metrics = score_alphas(
        sre,
        ND_ALPHAS,
        Path("/root/data"),
        Path("/root/data/synthetic/seed_split.json"),
        subsets=("all", "train", "holdout"),
    )
    metrics["calibrated_alpha"] = best_alpha
    metrics["calibration"] = calibration

    out_dir = Path("/results") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (out_dir / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")

    results.commit()
    emb_cache.commit()
    hf_cache.commit()

    calibrated_key = f"section_alpha_{best_alpha:.1f}"
    summary = {
        "run_id": run_id,
        "calibrated_alpha": best_alpha,
        "selection_accuracy": calibration["selection_stats"]["selection_accuracy"],
        "arms": {
            name: {
                "pair_auc": arm["all"]["pair"]["roc_auc"],
                "hard_neg_auc": arm["all"]["slices"]["neg_hardness"]["hard"]["pair_auc"],
                "mrr": arm["all"]["retrieval"]["mrr"],
                "recall@1": arm["all"]["retrieval"]["recall@1"],
                "recall@10": arm["all"]["retrieval"]["recall@10"],
            }
            for name, arm in metrics["arms"].items()
        },
        "calibrated_arm": calibrated_key,
    }
    return summary


@app.local_entrypoint()
def main(
    run_id: str = "nds_001",
    batch_size: int = 4,
    max_length: int = 4096,
    truncate_dim: int = 1024,
) -> None:
    res = run_remote.remote(
        run_id=run_id, batch_size=batch_size, max_length=max_length, truncate_dim=truncate_dim
    )
    print(f"\n=== nomad_drift_sectioned ({res['run_id']}) ===")
    print(f"calibrated alpha (fit on rrf_003) = {res['calibrated_alpha']}")
    print(f"section selection accuracy (rrf_003, vs ground truth) = {res['selection_accuracy']}")
    print(f"\n{'arm':22s} {'pairAUC':>8s} {'hardneg':>8s} {'MRR':>8s} {'R@1':>7s} {'R@10':>7s}")
    for name, s in res["arms"].items():
        marker = "  <-- calibrated" if name == res["calibrated_arm"] else ""
        print(
            f"{name:22s} {s['pair_auc']:8.4f} {s['hard_neg_auc']:8.4f} {s['mrr']:8.4f} "
            f"{s['recall@1']:7.4f} {s['recall@10']:7.4f}{marker}"
        )
    print(
        f"\nPull with:\n  modal volume get {RESULTS_VOLUME} {run_id} "
        f"./artifacts/nomad_drift_sectioned/{run_id}"
    )
