"""Modal GPU entrypoint for the hybrid TF-IDF + seeker-sectioned Voyage-4-nano baseline.

Encodes the (small) new seeker-sectioned embeddings on GPU, reusing the
existing cached candidate/corpus embeddings (unaffected by sectioning) that
are shipped into the image rather than re-encoded.

Usage:
  modal run baselines/hybrid_tfidf_voyage_seeker_sectioned/modal_eval.py
  modal volume get dorby-hybrid-seeker-sectioned-eval holdout ./artifacts/hybrid_tfidf_voyage_seeker_sectioned_modal/holdout
"""

from __future__ import annotations

import modal

APP_NAME = "dorby-hybrid-tfidf-voyage-seeker-sectioned"
RESULTS_VOLUME = "dorby-hybrid-seeker-sectioned-eval"
HF_CACHE_VOLUME = "dorby-twotower-hf-cache"  # reuse the existing HF model cache

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.51,<5",
        "sentence-transformers>=3.4.1,<6",
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
    # Cached candidate/corpus embeddings — unaffected by seeker-sectioning,
    # reused rather than re-encoded (see run_eval()'s fit_candidate_cache /
    # voyage_holdout_dir args).
    .add_local_dir(
        "artifacts/voyage_nano_holdout",
        remote_path="/root/artifacts/voyage_nano_holdout",
    )
    .add_local_dir(
        "artifacts/hybrid_tfidf_voyage_holdout/fit_cache/voyage",
        remote_path="/root/artifacts/hybrid_tfidf_voyage_holdout/fit_cache/voyage",
    )
)

results = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=30 * 60,
    volumes={
        "/results": results,
        "/cache/huggingface": hf_cache,
    },
)
def eval_remote(
    run_id: str = "holdout",
    fusion_mode: str = "alpha",
    retrieval_mode: str = "rrf",
    model: str = "voyageai/voyage-4-nano",
    batch_size: int = 4,
    max_length: int = 4096,
    truncate_dim: int = 1024,
) -> dict:
    import json
    from pathlib import Path

    from baselines.hybrid_tfidf_voyage_seeker_sectioned.eval import run_eval

    artifacts_dir = Path("/results") / run_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    metrics = run_eval(
        data_dir=Path("/root/data"),
        split_path=Path("/root/data/synthetic/seed_split.json"),
        artifacts_dir=artifacts_dir,
        fusion_mode=fusion_mode,
        retrieval_mode=retrieval_mode,
        voyage_model=model,
        batch_size=batch_size,
        max_length=max_length,
        truncate_dim=truncate_dim,
        voyage_holdout_dir=Path("/root/artifacts/voyage_nano_holdout"),
        fit_candidate_cache=Path(
            "/root/artifacts/hybrid_tfidf_voyage_holdout/fit_cache/voyage/emb_fit_pairs_cand.npy"
        ),
    )
    (artifacts_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    results.commit()
    return {
        "run_id": run_id,
        "fusion": metrics["fusion"],
        "solo_holdout": metrics["solo_holdout"],
        "pair": metrics["pair"],
        "retrieval": metrics["retrieval"],
    }


@app.local_entrypoint()
def main(
    run_id: str = "holdout",
    fusion_mode: str = "alpha",
    retrieval_mode: str = "rrf",
    model: str = "voyageai/voyage-4-nano",
    batch_size: int = 4,
    max_length: int = 4096,
    truncate_dim: int = 1024,
) -> None:
    """CLI: modal run baselines/hybrid_tfidf_voyage_seeker_sectioned/modal_eval.py"""
    result = eval_remote.remote(
        run_id=run_id,
        fusion_mode=fusion_mode,
        retrieval_mode=retrieval_mode,
        model=model,
        batch_size=batch_size,
        max_length=max_length,
        truncate_dim=truncate_dim,
    )
    print("=== Modal hybrid seeker-sectioned eval finished ===")
    print(f"run_id: {result['run_id']}")
    print("fusion:", result["fusion"])
    print("solo_holdout:", result["solo_holdout"])
    print("pair:", result["pair"])
    print("retrieval:", result["retrieval"])
    print(
        f"\nPull full metrics with:\n"
        f"  modal volume get {RESULTS_VOLUME} {result['run_id']} "
        f"./artifacts/hybrid_tfidf_voyage_seeker_sectioned_modal/{result['run_id']}"
    )
