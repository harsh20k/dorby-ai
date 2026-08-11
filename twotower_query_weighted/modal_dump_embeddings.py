"""Modal entrypoint: dump raw frozen + fine-tuned vectors for the topology graph.

    modal run twotower_query_weighted/modal_dump_embeddings.py
    modal volume get dorby-twotower-query-weighted-results embeddings_dump.json \\
        ./artifacts/twotower_query_weighted/embeddings_dump.json

Reuses the ``dorby-query-weighted-cache`` volume for the frozen encoder — the
200 seeker/query texts and 178 candidate texts were already embedded by
``query_weighted/modal_eval.py``'s ``qw_001`` run, so this is a cache hit, not
a re-encode.
"""

from __future__ import annotations

import modal

APP_NAME = "dorby-twotower-query-weighted"
RESULTS_VOLUME = "dorby-twotower-query-weighted-results"
EMB_CACHE_VOLUME = "dorby-query-weighted-cache"  # shared read-only reuse
HF_CACHE_VOLUME = "dorby-twotower-hf-cache"

app = modal.App(APP_NAME)

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
        "twotower_query_weighted", "query_weighted", "eval_real_full", "twotower", "baselines", "synth_pipeline"
    )
    .add_local_dir("eval_real_full/data_frozen", remote_path="/root/eval_real_full/data_frozen")
    .add_local_dir("data", remote_path="/root/data")
    .add_local_dir("artifacts/twotower_top1_optimised", remote_path="/root/top1_runs")
)

results = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True)
emb_cache = modal.Volume.from_name(EMB_CACHE_VOLUME, create_if_missing=True)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True)


@app.function(
    image=image,
    gpu="L4",
    timeout=60 * 60,
    volumes={"/results": results, "/embcache": emb_cache, "/cache/huggingface": hf_cache},
)
def dump_remote(batch_size: int = 4) -> str:
    from pathlib import Path

    from baselines.voyage_nano.encode import VoyageNanoEncoder

    from twotower_query_weighted.dump_embeddings import build_dump, write_dump
    from twotower_query_weighted.eval import load_adapter_model

    frozen_encoder = VoyageNanoEncoder(
        model_name="voyageai/voyage-4-nano",
        device="cuda",
        max_length=4096,
        truncate_dim=1024,
        cache_dir=Path("/embcache"),
    )
    finetuned_model = load_adapter_model(
        model_name="voyageai/voyage-4-nano",
        adapter_dir=Path("/root/top1_runs/top1_ctrl_001/adapter"),
        device="cuda",
        max_length=4096,
        truncate_dim=1024,
    )

    dump = build_dump(
        frozen_encoder,
        finetuned_model,
        data_dir=Path("/root/data"),
        split_path=Path("/root/data/synthetic/seed_split.json"),
        batch_size=batch_size,
    )
    out = write_dump(dump, Path("/results/embeddings_dump.json"))
    results.commit()
    emb_cache.commit()
    hf_cache.commit()
    return str(out)


@app.local_entrypoint()
def main(batch_size: int = 4) -> None:
    path = dump_remote.remote(batch_size)
    print(f"done: {path}")
    print(
        f"\nPull with:\n  modal volume get {RESULTS_VOLUME} embeddings_dump.json "
        f"./artifacts/twotower_query_weighted/embeddings_dump.json"
    )
