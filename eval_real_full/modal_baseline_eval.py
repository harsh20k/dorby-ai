"""Modal entrypoint: score the frozen baselines on all 200 real pairs.

Companion to ``modal_eval.py`` (which handles the twotower adapters). This one
runs the *baseline* encoders — TF-IDF, frozen BERT, and the open-weight HF
embedders — through ``eval_real_full.baseline_eval``, so their numbers land on
the same 200-pair population as everything else in this experiment.

The three GPU images mirror ``baselines/hf_embedding/modal_eval.py`` exactly,
including the two models that need a non-default loading path (NV-Embed-v2's
pinned older transformers, BGE-en-ICL's FlagEmbedding library). They are
duplicated rather than imported because that module builds its images at import
time against its own local-source list; importing it would couple this
experiment's runs to edits made for that baseline.

    modal run eval_real_full/modal_baseline_eval.py --configs tfidf,bert
    modal run eval_real_full/modal_baseline_eval.py --configs qwen8b --gpu A100-40GB
    modal volume get dorby-eval-real-full-results real200_baselines \\
        ./artifacts/eval_real_full/real200_baselines

Cost note: an 8B model at 8192 tokens over the 578 texts of the ``all`` subset
is the dominant expense. ``--subsets all,holdout`` is the default for that
reason — ``holdout`` doubles as the correctness gate (it must reproduce the
published row in docs/baseline-results-holdout.md), while ``train`` adds a
population nothing is being decided on.
"""

from __future__ import annotations

import modal

APP_NAME = "dorby-eval-real-full-baselines"
RESULTS_VOLUME = "dorby-eval-real-full-results"  # shared with modal_eval.py
HF_CACHE_VOLUME = "dorby-twotower-hf-cache"  # already holds every model tested

app = modal.App(APP_NAME)

_PACKAGES = ("eval_real_full", "baselines", "twotower", "synth_pipeline")
_ENV = {
    "HF_HOME": "/cache/huggingface",
    "TRANSFORMERS_CACHE": "/cache/huggingface",
    "TOKENIZERS_PARALLELISM": "false",
}


def _with_sources(img: modal.Image) -> modal.Image:
    """Mount code + data. The frozen manifest is JSON, so add_local_python_source
    (which ships only .py) would miss it and data.py's verify would fail."""
    return (
        img.env(_ENV)
        .add_local_python_source(*_PACKAGES)
        .add_local_dir("data", remote_path="/root/data")
        .add_local_dir(
            "eval_real_full/data_frozen", remote_path="/root/eval_real_full/data_frozen"
        )
    )


image = _with_sources(
    modal.Image.debian_slim(python_version="3.11").pip_install(
        "torch",
        "transformers>=4.51,<5",
        "sentence-transformers>=3.4.1,<6",
        "accelerate>=0.30",
        # Pinned, not floated: TfidfVectorizer's fitted vocabulary is not stable
        # across scikit-learn versions. On an unpinned image TF-IDF scored
        # holdout AUC 0.5828 where the published baseline (and this repo's venv,
        # sklearn 1.9.0) give 0.5922 — same code, same data. Every other model
        # here uses sklearn only for metric computation, which is stable, so the
        # pin costs nothing and makes the lexical baseline reproducible.
        "scikit-learn==1.9.0",
        "numpy>=1.26.0",
        "tqdm>=4.66.0",
        "datasets>=2.19",
        "einops",
    )
)

legacy_transformers_image = _with_sources(
    modal.Image.debian_slim(python_version="3.11").pip_install(
        "torch",
        "transformers==4.44.2",
        "sentence-transformers==3.0.1",
        "accelerate>=0.30",
        # Pinned, not floated: TfidfVectorizer's fitted vocabulary is not stable
        # across scikit-learn versions. On an unpinned image TF-IDF scored
        # holdout AUC 0.5828 where the published baseline (and this repo's venv,
        # sklearn 1.9.0) give 0.5922 — same code, same data. Every other model
        # here uses sklearn only for metric computation, which is stable, so the
        # pin costs nothing and makes the lexical baseline reproducible.
        "scikit-learn==1.9.0",
        "numpy>=1.26.0",
        "tqdm>=4.66.0",
        "datasets>=2.19",
        "einops",
    )
)

flagembedding_image = _with_sources(
    modal.Image.debian_slim(python_version="3.11").pip_install(
        "torch",
        "transformers>=4.51,<5",
        "FlagEmbedding",
        "accelerate>=0.30",
        # Pinned, not floated: TfidfVectorizer's fitted vocabulary is not stable
        # across scikit-learn versions. On an unpinned image TF-IDF scored
        # holdout AUC 0.5828 where the published baseline (and this repo's venv,
        # sklearn 1.9.0) give 0.5922 — same code, same data. Every other model
        # here uses sklearn only for metric computation, which is stable, so the
        # pin costs nothing and makes the lexical baseline reproducible.
        "scikit-learn==1.9.0",
        "numpy>=1.26.0",
        "tqdm>=4.66.0",
    )
)

results = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True)

# label -> what to score. `max_length` MUST match each model's published
# baseline run — that is the number the holdout subset has to reproduce, and a
# shorter context silently changes it. Every published HF run used 8192 with no
# Matryoshka truncation (verified against the metrics.json in the
# dorby-hf-embedding-eval volume); an earlier 4096 here made E5-Mistral and
# NV-Embed-v2 non-comparable to their own published rows.
CONFIGS: dict[str, dict] = {
    "tfidf": {"kind": "tfidf", "model": None, "device": "cpu", "max_length": 0},
    "bert": {
        "kind": "bert",
        "model": "bert-base-uncased",
        "device": "cuda",
        "max_length": 512,
        "batch_size": 16,
    },
    "qwen8b": {
        "kind": "hf",
        "model": "Qwen/Qwen3-Embedding-8B",
        "device": "cuda",
        "max_length": 8192,
        "batch_size": 2,
    },
    "e5_mistral": {
        "kind": "hf",
        "model": "intfloat/e5-mistral-7b-instruct",
        "device": "cuda",
        "max_length": 8192,
        "batch_size": 2,
    },
    "zembed": {
        "kind": "hf",
        "model": "zeroentropy/zembed-1-embedding",
        "device": "cuda",
        "max_length": 8192,
        "batch_size": 2,
    },
    "bge_en_icl": {
        "kind": "hf",
        "model": "BAAI/bge-en-icl",
        "device": "cuda",
        "max_length": 8192,
        "batch_size": 2,
    },
    "nv_embed": {
        "kind": "hf",
        "model": "nvidia/NV-Embed-v2",
        "device": "cuda",
        "max_length": 8192,
        "batch_size": 2,
    },
}


def _library_versions() -> dict:
    """Version provenance for the libraries that can move a metric."""
    import importlib

    out = {}
    for mod in ("sklearn", "numpy", "torch", "transformers", "sentence_transformers"):
        try:
            out[mod] = importlib.import_module(mod).__version__
        except Exception:  # not installed in this image — legitimate for some
            out[mod] = None
    return out


def _run_one(run_id: str, config: str, subsets: str, commit_hf_cache: bool = True) -> dict:
    from pathlib import Path

    from eval_real_full.baseline_eval import run_baseline_eval, write_metrics

    spec = CONFIGS[config]
    metrics = run_baseline_eval(
        kind=spec["kind"],
        data_dir=Path("/root/data"),
        split_path=Path("/root/data/synthetic/seed_split.json"),
        label=config,
        model_name=spec["model"],
        subsets=tuple(s.strip() for s in subsets.split(",") if s.strip()),
        batch_size=spec.get("batch_size", 8),
        max_length=spec.get("max_length", 8192),
        truncate_dim=spec.get("truncate_dim"),
        dtype=spec.get("dtype", "auto"),
        device=spec["device"],
        # Embedding cache is scratch: each subset's texts differ, keys are
        # content-hashed, and nothing here is reused across runs.
        cache_dir=Path("/tmp") / f"cache_{config}",
    )
    # Recorded because TF-IDF's numbers proved version-dependent (see the
    # scikit-learn pin above) — without this the discrepancy is invisible.
    metrics["library_versions"] = _library_versions()
    write_metrics(metrics, Path("/results") / run_id / config)
    results.commit()
    # The CPU function has no HF cache mounted (TF-IDF downloads nothing), and
    # committing an unattached volume is a hard error.
    if commit_hf_cache:
        hf_cache.commit()
    return {
        "config": config,
        "summary": {
            subset: {
                "pair_auc": m["pair"]["roc_auc"],
                "hard_neg_auc": m["slices"]["neg_hardness"]["hard"]["pair_auc"],
                "mrr": m["retrieval"]["mrr"],
                "recall@1": m["retrieval"]["recall@1"],
                "recall@10": m["retrieval"]["recall@10"],
                "n_candidates": m["n_candidates"],
            }
            for subset, m in metrics["subsets"].items()
        },
    }


@app.function(image=image, timeout=60 * 60, volumes={"/results": results})
def eval_cpu(run_id: str, config: str, subsets: str) -> dict:
    """TF-IDF has no model to place on a device — no GPU, so near-zero cost."""
    return _run_one(run_id, config, subsets, commit_hf_cache=False)


@app.function(
    image=image,
    gpu="A10G",
    timeout=90 * 60,
    volumes={"/results": results, "/cache/huggingface": hf_cache},
)
def eval_gpu(run_id: str, config: str, subsets: str) -> dict:
    return _run_one(run_id, config, subsets)


@app.function(
    image=legacy_transformers_image,
    gpu="A10G",
    timeout=90 * 60,
    volumes={"/results": results, "/cache/huggingface": hf_cache},
)
def eval_gpu_legacy(run_id: str, config: str, subsets: str) -> dict:
    return _run_one(run_id, config, subsets)


@app.function(
    image=flagembedding_image,
    gpu="A10G",
    timeout=90 * 60,
    volumes={"/results": results, "/cache/huggingface": hf_cache},
)
def eval_gpu_flag(run_id: str, config: str, subsets: str) -> dict:
    return _run_one(run_id, config, subsets)


@app.local_entrypoint()
def main(
    run_id: str = "real200_baselines",
    configs: str = "tfidf,bert",
    subsets: str = "all,holdout",
    gpu: str = "A10G",
) -> None:
    """Run each config in parallel. ``--gpu`` applies to the GPU configs only."""
    from baselines.hf_embedding.models import get_model_spec

    wanted = [c.strip() for c in configs.split(",") if c.strip()]
    unknown = [c for c in wanted if c not in CONFIGS]
    if unknown:
        raise SystemExit(f"unknown configs {unknown}; choices: {list(CONFIGS)}")

    print(f"run_id={run_id} configs={wanted} subsets={subsets} gpu={gpu}")
    handles = []
    for c in wanted:
        spec = CONFIGS[c]
        if spec["kind"] == "tfidf":
            fn = eval_cpu
        else:
            model_spec = get_model_spec(spec["model"]) if spec["model"] else None
            if model_spec is not None and model_spec.loader == "flagembedding_icl":
                fn = eval_gpu_flag
            elif model_spec is not None and model_spec.requires_legacy_transformers:
                fn = eval_gpu_legacy
            else:
                fn = eval_gpu
            if gpu and gpu != "A10G":
                fn = fn.with_options(gpu=gpu)
        handles.append(fn.spawn(run_id, c, subsets))

    for h in handles:
        res = h.get()
        print(f"\n=== {res['config']} ===")
        for subset, s in res["summary"].items():
            print(
                f"  {subset:8s} corpus={s['n_candidates']:4d} "
                f"AUC={s['pair_auc']:.4f} hard={s['hard_neg_auc']:.4f} "
                f"MRR={s['mrr']:.4f} R@1={s['recall@1']:.4f} R@10={s['recall@10']:.4f}"
            )
    print(
        f"\nPull with:\n  modal volume get {RESULTS_VOLUME} {run_id} "
        f"./artifacts/eval_real_full/{run_id}"
    )
