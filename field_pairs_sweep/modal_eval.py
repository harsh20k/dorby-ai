"""Modal entrypoint for the identity-field-pair arms (frozen voyage-4-nano).

Inference only, L4 like ``query_weighted``.

    modal run field_pairs_sweep/modal_eval.py
    modal volume get dorby-field-pairs-sweep-results fp_001 \\
        ./artifacts/field_pairs_sweep/fp_001

Mounts ``query_weighted``'s own embedding-cache volume read/write: candidate
text is identical to every other experiment in this project, so that side is
a free cache hit; only the three two-field seeker arms (600 new texts total)
cost a fresh encode. Own results volume, own app, per isolation.

Settings pinned to ``max_length=4096`` / ``truncate_dim=1024`` to match every
other frozen-nano row in this project.
"""

from __future__ import annotations

import modal

APP_NAME = "dorby-field-pairs-sweep"
RESULTS_VOLUME = "dorby-field-pairs-sweep-results"
EMB_CACHE_VOLUME = "dorby-query-weighted-cache"  # shared read/write, free hits on candidate texts
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
    .add_local_python_source("field_pairs_sweep", "eval_real_full", "baselines", "twotower", "synth_pipeline")
    .add_local_dir("data", remote_path="/root/data")
    .add_local_dir(
        "eval_real_full/data_frozen", remote_path="/root/eval_real_full/data_frozen"
    )
)

results = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True)
emb_cache = modal.Volume.from_name(EMB_CACHE_VOLUME, create_if_missing=True)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU,
    timeout=60 * 60,
    volumes={
        "/results": results,
        "/embcache": emb_cache,
        "/cache/huggingface": hf_cache,
    },
)
def eval_remote(
    run_id: str = "fp_001",
    subsets: str = "all,train,holdout",
    batch_size: int = 4,
    max_length: int = 4096,
    truncate_dim: int = 1024,
) -> dict:
    from pathlib import Path

    from baselines.voyage_nano.encode import VoyageNanoEncoder

    from field_pairs_sweep.eval import run_all_arms, write_metrics

    encoder = VoyageNanoEncoder(
        model_name="voyageai/voyage-4-nano",
        device="cuda",
        max_length=max_length,
        truncate_dim=truncate_dim,
        cache_dir=Path("/embcache"),
    )
    metrics = run_all_arms(
        encoder,
        Path("/root/data"),
        Path("/root/data/synthetic/seed_split.json"),
        subsets=tuple(s.strip() for s in subsets.split(",") if s.strip()),
        batch_size=batch_size,
    )
    write_metrics(metrics, Path("/results") / run_id)
    results.commit()
    emb_cache.commit()
    hf_cache.commit()

    return {
        "run_id": run_id,
        "summary": {
            name: {
                "pair_auc": arm["all"]["pair"]["roc_auc"],
                "hard_neg_auc": arm["all"]["slices"]["neg_hardness"]["hard"]["pair_auc"],
                "mrr": arm["all"]["retrieval"]["mrr"],
                "recall@1": arm["all"]["retrieval"]["recall@1"],
                "recall@5": arm["all"]["retrieval"]["recall@5"],
                "recall@10": arm["all"]["retrieval"]["recall@10"],
            }
            for name, arm in metrics["arms"].items()
        },
    }


@app.local_entrypoint()
def main(
    run_id: str = "fp_001",
    subsets: str = "all,train,holdout",
    batch_size: int = 4,
    max_length: int = 4096,
    truncate_dim: int = 1024,
) -> None:
    res = eval_remote.remote(
        run_id=run_id,
        subsets=subsets,
        batch_size=batch_size,
        max_length=max_length,
        truncate_dim=truncate_dim,
    )
    print(f"\n=== field-pair identity arms, all 200 real pairs ({res['run_id']}) ===")
    print(f"{'arm':22s} {'pairAUC':>8s} {'hardneg':>8s} {'MRR':>8s} {'R@1':>7s} {'R@5':>7s} {'R@10':>7s}")
    for name, s in res["summary"].items():
        print(
            f"{name:22s} {s['pair_auc']:8.4f} {s['hard_neg_auc']:8.4f} {s['mrr']:8.4f} "
            f"{s['recall@1']:7.4f} {s['recall@5']:7.4f} {s['recall@10']:7.4f}"
        )
    print(
        f"\nPull with:\n  modal volume get {RESULTS_VOLUME} {run_id} "
        f"./artifacts/field_pairs_sweep/{run_id}"
    )
