"""Modal GPU entrypoint: score the split two-tower adapters on all 200 real
pairs. Query tower encodes each pair's real ``searchQuery``; candidate tower
encodes ``candidate_to_text`` — matching training exactly, so no eval/train
mismatch bug (docs/possible-bugs.md #5) is possible here by construction.

    modal run twotower_split/modal_eval.py --run-id split_001
    modal volume get dorby-twotower-split-eval-results split_001 \\
        ./artifacts/twotower_split/split_001_real200
"""

from __future__ import annotations

import modal

APP_NAME = "dorby-twotower-split"
RESULTS_VOLUME = "dorby-twotower-split-eval-results"
CHECKPOINT_VOLUME = "dorby-twotower-split-checkpoints"
HF_CACHE_VOLUME = "dorby-twotower-hf-cache"

app = modal.App(APP_NAME)
GPU = "L4"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.51,<5",
        "sentence-transformers>=5.6,<6",
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
        "twotower_split", "query_weighted", "eval_real_full", "twotower", "baselines", "synth_pipeline"
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
    volumes={"/results": results, "/checkpoints": checkpoints, "/cache/huggingface": hf_cache},
)
def eval_remote(run_id: str, batch_size: int = 8) -> dict:
    from pathlib import Path

    import numpy as np
    from sentence_transformers import SentenceTransformer

    from baselines.metrics import pair_metrics, retrieval_metrics, slice_metrics
    from baselines.voyage_nano.encode import cosine_scores, l2_normalize
    from eval_real_full.data import load_real_pairs
    from query_weighted import text as qtext
    from twotower_split.config import build_config

    cfg = build_config(run_id=run_id)
    ckpt_dir = Path("/checkpoints") / run_id / "best"

    def load_tower(adapter_dir: Path) -> SentenceTransformer:
        model = SentenceTransformer(
            cfg.model_name, trust_remote_code=True, truncate_dim=cfg.truncate_dim, device="cuda"
        )
        model.max_seq_length = cfg.max_seq_length
        model.load_adapter(str(adapter_dir))
        return model

    query_model = load_tower(ckpt_dir / "query_adapter")
    doc_model = load_tower(ckpt_dir / "doc_adapter")

    def encode(model, texts, prompt, role):
        raw = model.encode(
            [prompt + t for t in texts], batch_size=batch_size, show_progress_bar=False,
            normalize_embeddings=True, convert_to_numpy=True,
        )
        return l2_normalize(np.asarray(raw, dtype=np.float32))

    data_dir, split_path = Path("/root/data"), Path("/root/data/synthetic/seed_split.json")
    out = {"label": run_id, "subsets": {}}
    for subset in ("all", "train", "holdout"):
        ps = load_real_pairs(data_dir, split_path, subset=subset, verify=True)
        positives = [p.pair for p in ps.pairs if p.label == "pos"]
        negatives = [p.pair for p in ps.pairs if p.label == "neg"]

        pos_seeker_texts = [qtext.query_only(r["userContactFile"], r["searchQuery"]) for r in positives]
        neg_seeker_texts = [qtext.query_only(r["userContactFile"], r["searchQuery"]) for r in negatives]
        pos_cand_texts = [qtext.candidate_to_text(r["matchContactFile"]) for r in positives]
        neg_cand_texts = [qtext.candidate_to_text(r["matchContactFile"]) for r in negatives]

        corpus_ids, corpus_texts, seen = [], [], set()
        for r in positives + negatives:
            mid = r["matchContactId"]
            if mid in seen:
                continue
            seen.add(mid)
            corpus_ids.append(mid)
            corpus_texts.append(qtext.candidate_to_text(r["matchContactFile"]))

        pos_seeker_emb = encode(query_model, pos_seeker_texts, cfg.query_prompt, "query")
        neg_seeker_emb = encode(query_model, neg_seeker_texts, cfg.query_prompt, "query")
        pos_cand_emb = encode(doc_model, pos_cand_texts, cfg.document_prompt, "document")
        neg_cand_emb = encode(doc_model, neg_cand_texts, cfg.document_prompt, "document")
        corpus_emb = encode(doc_model, corpus_texts, cfg.document_prompt, "document")

        pos_scores = cosine_scores(pos_seeker_emb, pos_cand_emb)
        neg_scores = cosine_scores(neg_seeker_emb, neg_cand_emb)
        pos_target_ids = [r["matchContactId"] for r in positives]

        metrics = {
            "n_candidates": len(corpus_ids),
            "pair": pair_metrics(pos_scores, neg_scores),
            "retrieval": retrieval_metrics(
                query_embs=pos_seeker_emb, target_ids=pos_target_ids,
                candidate_ids=corpus_ids, candidate_embs=corpus_emb,
            ),
            "slices": slice_metrics(
                positives=positives, negatives=negatives,
                pos_scores=pos_scores, neg_scores=neg_scores,
                neg_seeker_texts=neg_seeker_texts, neg_cand_texts=neg_cand_texts,
                query_embs=pos_seeker_emb, target_ids=pos_target_ids,
                candidate_ids=corpus_ids, candidate_embs=corpus_emb,
            ),
        }
        out["subsets"][subset] = metrics
        m = metrics
        print(
            f"{subset:8s} corpus={m['n_candidates']:4d} AUC={m['pair']['roc_auc']:.4f} "
            f"hard={m['slices']['neg_hardness']['hard']['pair_auc']:.4f} "
            f"MRR={m['retrieval']['mrr']:.4f} R@1={m['retrieval']['recall@1']:.4f} "
            f"R@10={m['retrieval']['recall@10']:.4f}"
        )

    import json
    out_path = Path("/results") / run_id / "metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    results.commit()
    hf_cache.commit()
    a = out["subsets"]["all"]
    return {
        "pair_auc": a["pair"]["roc_auc"],
        "hard_neg_auc": a["slices"]["neg_hardness"]["hard"]["pair_auc"],
        "mrr": a["retrieval"]["mrr"],
        "recall@1": a["retrieval"]["recall@1"],
        "recall@10": a["retrieval"]["recall@10"],
    }


@app.local_entrypoint()
def main(run_id: str = "split_001", batch_size: int = 8) -> None:
    summary = eval_remote.remote(run_id, batch_size)
    print(f"\n=== {run_id} split two-tower, all 200 real pairs ===")
    print(
        f"AUC={summary['pair_auc']:.4f} hard={summary['hard_neg_auc']:.4f} "
        f"MRR={summary['mrr']:.4f} R@1={summary['recall@1']:.4f} R@10={summary['recall@10']:.4f}"
    )
    print(
        f"\nPull with:\n  modal volume get {RESULTS_VOLUME} {run_id} "
        f"./artifacts/twotower_split/{run_id}_real200"
    )
