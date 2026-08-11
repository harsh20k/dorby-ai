"""Modal GPU entrypoint: score query_only_001 on seeker text that matches its
training distribution (query only, no profile).

Built matched-distribution from the start — the mismatch bug in
`twotower_no_query/`'s first eval pass (scored on text the model never
trained on; `docs/possible-bugs.md` #5) is not repeated here. Reuses
`twotower_query_weighted.eval`'s already-published `query_only` scoring path
read-only, applied to the `query_only_001` adapter — same function that
scored `top1_ctrl`'s `query_only` arm (the eval-time swap, R@1 0.32), so the
two are directly comparable.

    modal run twotower_query_only/modal_eval_matched.py --run-id query_only_001
    modal volume get dorby-twotower-query-only-eval-results query_only_001_matched \\
        ./artifacts/twotower_query_only/query_only_001_matched
"""

from __future__ import annotations

import modal

APP_NAME = "dorby-twotower-query-only"
RESULTS_VOLUME = "dorby-twotower-query-only-eval-results"
CHECKPOINT_VOLUME = "dorby-twotower-query-only-checkpoints"
HF_CACHE_VOLUME = "dorby-twotower-hf-cache"

app = modal.App(APP_NAME)

GPU = "L4"

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
    .add_local_python_source(
        "twotower_query_only", "twotower_query_weighted", "query_weighted",
        "eval_real_full", "twotower", "baselines", "synth_pipeline",
    )
    .add_local_dir("eval_real_full/data_frozen", remote_path="/root/eval_real_full/data_frozen")
    .add_local_dir("data", remote_path="/root/data")
)

results = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True)
checkpoints = modal.Volume.from_name(CHECKPOINT_VOLUME, create_if_missing=True)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU,
    timeout=60 * 60,
    volumes={
        "/results": results,
        "/checkpoints": checkpoints,
        "/cache/huggingface": hf_cache,
    },
)
def eval_remote(run_id: str, batch_size: int = 8) -> dict:
    from pathlib import Path

    from twotower_query_weighted.eval import (
        encode_everything,
        load_adapter_model,
        score_arm,
        write_metrics,
    )
    from eval_real_full.data import load_real_pairs

    model = load_adapter_model(
        model_name="voyageai/voyage-4-nano",
        adapter_dir=Path("/checkpoints") / run_id / "adapter",
        device="cuda",
        max_length=4096,
        truncate_dim=1024,
    )
    enc = encode_everything(model, Path("/root/data"), Path("/root/data/synthetic/seed_split.json"), batch_size=batch_size)

    out = {"label": f"{run_id}_matched", "arms": {}}
    for subset in ("all", "train", "holdout"):
        ps = load_real_pairs(Path("/root/data"), Path("/root/data/synthetic/seed_split.json"), subset=subset, verify=True)
        subset_ids = {p.pair_id for p in ps.pairs}
        out["arms"].setdefault("query_only", {})[subset] = score_arm(enc, enc.seeker["query_only"], subset_ids)

    write_metrics(out, Path("/results") / f"{run_id}_matched")
    results.commit()
    hf_cache.commit()

    a = out["arms"]["query_only"]["all"]
    return {
        "pair_auc": a["pair"]["roc_auc"],
        "hard_neg_auc": a["slices"]["neg_hardness"]["hard"]["pair_auc"],
        "mrr": a["retrieval"]["mrr"],
        "recall@1": a["retrieval"]["recall@1"],
        "recall@10": a["retrieval"]["recall@10"],
    }


@app.local_entrypoint()
def main(run_id: str = "query_only_001", batch_size: int = 8) -> None:
    summary = eval_remote.remote(run_id, batch_size)
    print(f"\n=== {run_id} scored on query_only text (matches training) ===")
    print(
        f"AUC={summary['pair_auc']:.4f} hard={summary['hard_neg_auc']:.4f} "
        f"MRR={summary['mrr']:.4f} R@1={summary['recall@1']:.4f} R@10={summary['recall@10']:.4f}"
    )
    print(
        f"\nPull with:\n  modal volume get {RESULTS_VOLUME} {run_id}_matched "
        f"./artifacts/twotower_query_only/{run_id}_matched"
    )
