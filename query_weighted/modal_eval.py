"""Modal entrypoint for the query-weighting arms (frozen voyage-4-nano).

Inference only, one small model, so this runs on an L4 rather than the A100 the
training packages need.

    modal run query_weighted/modal_eval.py
    modal run query_weighted/modal_eval.py --run-id qw_002 --subsets all,holdout
    modal volume get dorby-query-weighted-results qw_001 \\
        ./artifacts/query_weighted/qw_001

Embedding cache
---------------
``VoyageNanoEncoder`` keys its ``.npy`` cache by a SHA-256 of the *text list*
plus model/length/dim/role, so pointing ``cache_dir`` at a persistent volume
makes any re-run free for arms whose texts are unchanged. Adding a new α costs
nothing at all (pure arithmetic on cached vectors); adding a new text arm costs
one encode of 200 texts.

Settings are pinned to ``max_length=4096`` / ``truncate_dim=1024`` because those
are what produced the published frozen-nano all-200 row that the
``concat_baseline`` arm has to reproduce.

Truncation caveat: at 4096 tokens the mean seeker string (~2,500 tokens) fits,
but the longest do not. Front-loaded arms therefore keep the query and lose the
tail of the profile, while ``concat_baseline`` loses the query first. That is
part of what front-loading *does*, but it means the text-level arms confound
"more weight" with "protected from truncation" — the α arms do not, which is why
both families are run.
"""

from __future__ import annotations

import modal

APP_NAME = "dorby-query-weighted"
RESULTS_VOLUME = "dorby-query-weighted-results"
EMB_CACHE_VOLUME = "dorby-query-weighted-cache"
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
    .add_local_python_source("query_weighted", "eval_real_full", "baselines", "twotower", "synth_pipeline")
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
    timeout=120 * 60,
    volumes={
        "/results": results,
        "/embcache": emb_cache,
        "/cache/huggingface": hf_cache,
    },
)
def eval_remote(
    run_id: str = "qw_001",
    subsets: str = "all,train,holdout",
    batch_size: int = 4,
    max_length: int = 4096,
    truncate_dim: int = 1024,
) -> dict:
    from pathlib import Path

    from baselines.voyage_nano.encode import VoyageNanoEncoder

    from query_weighted.eval import run_all_arms, write_metrics

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
    run_id: str = "qw_001",
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
    print(f"\n=== query-weighted arms, all 200 real pairs ({res['run_id']}) ===")
    print(f"{'arm':18s} {'pairAUC':>8s} {'hardneg':>8s} {'MRR':>8s} {'R@1':>7s} {'R@5':>7s} {'R@10':>7s}")
    for name, s in res["summary"].items():
        print(
            f"{name:18s} {s['pair_auc']:8.4f} {s['hard_neg_auc']:8.4f} {s['mrr']:8.4f} "
            f"{s['recall@1']:7.4f} {s['recall@5']:7.4f} {s['recall@10']:7.4f}"
        )
    print(
        f"\nPull with:\n  modal volume get {RESULTS_VOLUME} {run_id} "
        f"./artifacts/query_weighted/{run_id}"
    )
