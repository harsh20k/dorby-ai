"""Modal GPU entrypoint: score no_query_001 on seeker text that MATCHES its
training distribution (profile only, no query) — not the query-included text
`eval_real_full.eval.run_eval` used in the first pass at this model.

Bug this exists to fix
-----------------------
`twotower.data.LabeledPair.seeker_text` (used by `twotower.eval.evaluate_pairs`,
and therefore by `eval_real_full.eval.run_eval`, and therefore by
`twotower_no_query/modal_eval.py`) is hardcoded to `seeker_to_text(profile,
searchQuery)` — profile *plus* query, always. That is correct for every other
adapter in this project, all of which were trained on that same text. It is
wrong for `no_query_001`, which was trained on `profile_to_text` alone: the
first eval pass fed it out-of-distribution input (query tokens it never saw
in training), so its all-200 recall@1 of 0.18 measured a train/eval mismatch,
not the quality of query-free training.

Neither `twotower/eval.py` nor `eval_real_full/eval.py` is edited — both are
shared, published-results code. Instead this reuses
`twotower_query_weighted.eval`'s already-published `profile_only` scoring path
(built for exactly this "seeker text without the query" case) read-only,
applied to the `no_query_001` adapter. Same function, same code that scored
`top1_ctrl`'s `profile_only` arm, so the two are directly comparable.

    modal run twotower_no_query/modal_eval_matched.py --run-id no_query_001
    modal volume get dorby-twotower-no-query-eval-results no_query_001_matched \\
        ./artifacts/twotower_no_query/no_query_001_matched
"""

from __future__ import annotations

import modal

APP_NAME = "dorby-twotower-no-query"
RESULTS_VOLUME = "dorby-twotower-no-query-eval-results"
CHECKPOINT_VOLUME = "dorby-twotower-no-query-checkpoints"
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
        "twotower_no_query", "twotower_query_weighted", "query_weighted",
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
        out["arms"].setdefault("profile_only", {})[subset] = score_arm(enc, enc.seeker["profile_only"], subset_ids)

    write_metrics(out, Path("/results") / f"{run_id}_matched")
    results.commit()
    hf_cache.commit()

    a = out["arms"]["profile_only"]["all"]
    return {
        "pair_auc": a["pair"]["roc_auc"],
        "hard_neg_auc": a["slices"]["neg_hardness"]["hard"]["pair_auc"],
        "mrr": a["retrieval"]["mrr"],
        "recall@1": a["retrieval"]["recall@1"],
        "recall@10": a["retrieval"]["recall@10"],
    }


@app.local_entrypoint()
def main(run_id: str = "no_query_001", batch_size: int = 8) -> None:
    summary = eval_remote.remote(run_id, batch_size)
    print(f"\n=== {run_id} scored on profile_only text (matches training) ===")
    print(
        f"AUC={summary['pair_auc']:.4f} hard={summary['hard_neg_auc']:.4f} "
        f"MRR={summary['mrr']:.4f} R@1={summary['recall@1']:.4f} R@10={summary['recall@10']:.4f}"
    )
    print(
        f"\nPull with:\n  modal volume get {RESULTS_VOLUME} {run_id}_matched "
        f"./artifacts/twotower_no_query/{run_id}_matched"
    )
