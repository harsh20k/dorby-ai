"""Late-fusion baseline: TF-IDF lexical cosine + Voyage-4-nano embedding cosine.

Fits a fusion head on the real (non-holdout) train pool, then evaluates once
on the frozen 69-pair holdout — same population as docs/baseline-results-holdout.md.

TF-IDF uses one vocab fitted on the real train pool (no holdout leak);
holdout is encoded with that same vocab. Voyage-4-nano holdout embeddings
are reused from ``artifacts/voyage_nano_holdout``. Fit-set voyage embeddings
are cached under ``--artifacts-dir``.

Usage:
  python -m baselines.hybrid_tfidf_voyage.eval \\
      --artifacts-dir artifacts/hybrid_tfidf_voyage_holdout
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np

from baselines.bert_frozen.text import candidate_to_text, seeker_to_text
from baselines.holdout import filter_to_holdout
from baselines.hybrid_tfidf_voyage.fusion import (
    FusionModel,
    fit_fusion,
    rrf_score_matrix,
)
from baselines.metrics import (
    pair_metrics,
    print_metrics,
    ranks_from_score_matrix,
    retrieval_metrics_from_ranks,
    slice_metrics,
)
from baselines.tfidf.encode import TfidfEncoder, cosine_scores as tfidf_cosine
from baselines.voyage_nano.encode import VoyageNanoEncoder, cosine_scores as voyage_cosine
from twotower.data import LabeledPair, build_split_bundle


def load_pairs(data_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pos_path = data_dir / "dataset_positive.json"
    neg_path = data_dir / "dataset_negative.json"
    if not pos_path.exists() or not neg_path.exists():
        raise FileNotFoundError(
            f"Expected {pos_path.name} and {neg_path.name} under {data_dir}"
        )
    return json.loads(pos_path.read_text()), json.loads(neg_path.read_text())


def build_candidate_corpus(
    positives: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    ids: list[str] = []
    texts: list[str] = []
    seen: set[str] = set()
    for record in positives + negatives:
        mid = record["matchContactId"]
        if mid in seen:
            continue
        seen.add(mid)
        ids.append(mid)
        texts.append(candidate_to_text(record["matchContactFile"]))
    return ids, texts


def _load_npy(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.load(path)


def _pair_score_arrays(
    pairs: list[LabeledPair],
    tfidf_enc: TfidfEncoder,
    voyage_enc: VoyageNanoEncoder,
    *,
    cache_prefix: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (tfidf_scores, voyage_scores, labels) for a list of LabeledPairs."""
    seeker_texts = [p.seeker_text for p in pairs]
    cand_texts = [p.candidate_text for p in pairs]
    labels = np.array([p.y for p in pairs], dtype=np.int32)

    tfidf_s = tfidf_enc.encode(seeker_texts, cache_name=f"{cache_prefix}_seeker")
    tfidf_c = tfidf_enc.encode(cand_texts, cache_name=f"{cache_prefix}_cand")
    voyage_s = voyage_enc.encode(
        seeker_texts,
        role="query",
        batch_size=batch_size,
        cache_name=f"{cache_prefix}_seeker",
    )
    voyage_c = voyage_enc.encode(
        cand_texts,
        role="document",
        batch_size=batch_size,
        cache_name=f"{cache_prefix}_cand",
    )
    return tfidf_cosine(tfidf_s, tfidf_c), voyage_cosine(voyage_s, voyage_c), labels


def _score_matrix(
    query_embs: np.ndarray, candidate_embs: np.ndarray
) -> np.ndarray:
    """(n_queries × n_candidates) cosine matrix for L2-normalized rows."""
    return query_embs @ candidate_embs.T


def run_eval(
    data_dir: Path,
    split_path: Path,
    artifacts_dir: Path,
    *,
    fusion_mode: Literal["alpha", "logistic"] = "alpha",
    retrieval_mode: Literal["score_fusion", "rrf"] = "rrf",
    rrf_k: int = 60,
    logistic_C: float = 1.0,
    max_features: int = 20000,
    ngram_range: tuple[int, int] = (1, 2),
    voyage_model: str = "voyageai/voyage-4-nano",
    batch_size: int = 4,
    max_length: int = 4096,
    truncate_dim: int = 1024,
    voyage_holdout_dir: Path = Path("artifacts/voyage_nano_holdout"),
    seed: int = 42,
) -> dict[str, Any]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    fit_cache = artifacts_dir / "fit_cache"
    fit_cache.mkdir(parents=True, exist_ok=True)

    # --- fit set: all real non-holdout pairs (train + train_dev), no synth ---
    bundle = build_split_bundle(
        data_dir,
        split_path,
        include_synth=False,
        seed=seed,
    )
    fit_pairs = sorted(bundle.train + bundle.train_dev, key=lambda p: p.pair_id)
    print(
        f"fit set: {len(fit_pairs)} real pairs "
        f"(train={len(bundle.train)}, train_dev={len(bundle.train_dev)}); "
        f"holdout={len(bundle.holdout)}"
    )

    # --- holdout pairs in the same order as the cached baseline evals ---
    positives, negatives = load_pairs(data_dir)
    positives, negatives = filter_to_holdout(positives, negatives, split_path)
    print(f"holdout: {len(positives)} pos / {len(negatives)} neg")

    pos_seeker_texts = [
        seeker_to_text(r["userContactFile"], r["searchQuery"]) for r in positives
    ]
    neg_seeker_texts = [
        seeker_to_text(r["userContactFile"], r["searchQuery"]) for r in negatives
    ]
    pos_cand_texts = [candidate_to_text(r["matchContactFile"]) for r in positives]
    neg_cand_texts = [candidate_to_text(r["matchContactFile"]) for r in negatives]
    corpus_ids, corpus_texts = build_candidate_corpus(positives, negatives)

    # --- encoders ---
    # One shared TF-IDF vocab fitted on the real train pool only (no holdout
    # texts). Holdout is encoded with that same vocab so fusion weights
    # transfer. Voyage holdout embeddings reuse the published nano cache
    # (frozen model → identical scores).
    tfidf_enc = TfidfEncoder(
        max_features=max_features,
        ngram_range=ngram_range,
        cache_dir=fit_cache / "tfidf",
    )
    fit_texts = [p.seeker_text for p in fit_pairs] + [
        p.candidate_text for p in fit_pairs
    ]
    tfidf_enc.fit(fit_texts)

    voyage_enc = VoyageNanoEncoder(
        model_name=voyage_model,
        max_length=max_length,
        truncate_dim=truncate_dim,
        cache_dir=fit_cache / "voyage",
    )

    fit_tfidf, fit_voyage, fit_y = _pair_score_arrays(
        fit_pairs,
        tfidf_enc,
        voyage_enc,
        cache_prefix="fit_pairs",
        batch_size=batch_size,
    )
    fusion: FusionModel = fit_fusion(
        fusion_mode,
        fit_tfidf,
        fit_voyage,
        fit_y,
        C=logistic_C,
        seed=seed,
    )
    print(f"fusion: {json.dumps(fusion.to_dict(), indent=2)}")

    # Holdout TF-IDF via train-pool vocab (consistent with fit scores).
    tfidf_pos_s = tfidf_enc.encode(pos_seeker_texts, cache_name="hold_pos_seeker")
    tfidf_neg_s = tfidf_enc.encode(neg_seeker_texts, cache_name="hold_neg_seeker")
    tfidf_pos_c = tfidf_enc.encode(pos_cand_texts, cache_name="hold_pos_cand")
    tfidf_neg_c = tfidf_enc.encode(neg_cand_texts, cache_name="hold_neg_cand")
    tfidf_corpus = tfidf_enc.encode(corpus_texts, cache_name="hold_corpus")

    # Holdout Voyage from published cache (same frozen encoder).
    voyage_pos_s = _load_npy(voyage_holdout_dir / "emb_pos_seeker.npy")
    voyage_neg_s = _load_npy(voyage_holdout_dir / "emb_neg_seeker.npy")
    voyage_pos_c = _load_npy(voyage_holdout_dir / "emb_pos_cand.npy")
    voyage_neg_c = _load_npy(voyage_holdout_dir / "emb_neg_cand.npy")
    voyage_corpus = _load_npy(voyage_holdout_dir / "emb_corpus.npy")

    if len(voyage_pos_s) != len(positives) or len(voyage_neg_s) != len(negatives):
        raise ValueError(
            f"voyage holdout cache size mismatch: pos={len(voyage_pos_s)}/"
            f"{len(positives)} neg={len(voyage_neg_s)}/{len(negatives)} — "
            "re-run baselines.voyage_nano.eval --holdout-only first"
        )
    if len(voyage_corpus) != len(corpus_ids):
        raise ValueError(
            f"voyage corpus cache size mismatch: {len(voyage_corpus)} vs "
            f"{len(corpus_ids)}"
        )

    hold_tfidf_pos = tfidf_cosine(tfidf_pos_s, tfidf_pos_c)
    hold_tfidf_neg = tfidf_cosine(tfidf_neg_s, tfidf_neg_c)
    hold_voyage_pos = voyage_cosine(voyage_pos_s, voyage_pos_c)
    hold_voyage_neg = voyage_cosine(voyage_neg_s, voyage_neg_c)

    pos_scores = fusion.pair_scores(hold_tfidf_pos, hold_voyage_pos)
    neg_scores = fusion.pair_scores(hold_tfidf_neg, hold_voyage_neg)
    pair = pair_metrics(pos_scores, neg_scores)

    # Retrieval: fuse Q×C score matrices (or RRF ranks)
    tfidf_qc = _score_matrix(tfidf_pos_s, tfidf_corpus)
    voyage_qc = _score_matrix(voyage_pos_s, voyage_corpus)
    if retrieval_mode == "rrf":
        fused_qc = rrf_score_matrix(tfidf_qc, voyage_qc, k=rrf_k)
    else:
        fused_qc = fusion.score_matrix(tfidf_qc, voyage_qc)

    pos_target_ids = [r["matchContactId"] for r in positives]
    ranks = ranks_from_score_matrix(fused_qc, pos_target_ids, corpus_ids)
    retrieval = retrieval_metrics_from_ranks(ranks)
    slices = slice_metrics(
        positives=positives,
        negatives=negatives,
        pos_scores=pos_scores,
        neg_scores=neg_scores,
        neg_seeker_texts=neg_seeker_texts,
        neg_cand_texts=neg_cand_texts,
        query_embs=None,
        target_ids=pos_target_ids,
        candidate_ids=corpus_ids,
        candidate_embs=None,
        ranks=ranks,
    )

    # Solo baselines on the same holdout caches (sanity / attribution)
    solo = {
        "tfidf_pair_auc": float(
            pair_metrics(hold_tfidf_pos, hold_tfidf_neg)["roc_auc"]
        ),
        "voyage_pair_auc": float(
            pair_metrics(hold_voyage_pos, hold_voyage_neg)["roc_auc"]
        ),
        "tfidf_mrr": float(
            retrieval_metrics_from_ranks(
                ranks_from_score_matrix(tfidf_qc, pos_target_ids, corpus_ids)
            )["mrr"]
        ),
        "voyage_mrr": float(
            retrieval_metrics_from_ranks(
                ranks_from_score_matrix(voyage_qc, pos_target_ids, corpus_ids)
            )["mrr"]
        ),
    }

    return {
        "model_name": f"hybrid_tfidf_voyage({fusion_mode},{retrieval_mode})",
        "device": "cpu+cached",
        "max_length": max_length,
        "truncate_dim": truncate_dim,
        "fusion": fusion.to_dict(),
        "retrieval_mode": retrieval_mode,
        "rrf_k": rrf_k if retrieval_mode == "rrf" else None,
        "fit": {
            "n_pairs": len(fit_pairs),
            "n_pos": int(fit_y.sum()),
            "n_neg": int(len(fit_y) - fit_y.sum()),
            "include_synth": False,
            "split_hash": bundle.split_hash,
        },
        "solo_holdout": solo,
        "pair": pair,
        "retrieval": retrieval,
        "slices": slices,
        # unused by print_metrics but keeps text lists available for debug
        "n_corpus": len(corpus_ids),
        "n_holdout_texts_checked": {
            "pos_seeker": len(pos_seeker_texts),
            "neg_seeker": len(neg_seeker_texts),
            "pos_cand": len(pos_cand_texts),
            "neg_cand": len(neg_cand_texts),
            "corpus": len(corpus_texts),
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TF-IDF + Voyage-4-nano late-fusion baseline eval"
    )
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument(
        "--split-path",
        type=Path,
        default=None,
        help="Defaults to <data-dir>/synthetic/seed_split.json",
    )
    p.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts/hybrid_tfidf_voyage_holdout"),
    )
    p.add_argument(
        "--holdout-only",
        action="store_true",
        default=True,
        help="Always true for this baseline (holdout is the only eval split).",
    )
    p.add_argument(
        "--fusion",
        choices=("alpha", "logistic"),
        default="alpha",
        help="How to combine TF-IDF and Voyage pair scores (default: alpha).",
    )
    p.add_argument(
        "--retrieval",
        choices=("score_fusion", "rrf"),
        default="rrf",
        help="Retrieval ranking: apply fusion to Q×C scores, or RRF ranks (default).",
    )
    p.add_argument("--rrf-k", type=int, default=60)
    p.add_argument("--logistic-C", type=float, default=1.0)
    p.add_argument("--max-features", type=int, default=20000)
    p.add_argument("--ngram-min", type=int, default=1)
    p.add_argument("--ngram-max", type=int, default=2)
    p.add_argument("--model", type=str, default="voyageai/voyage-4-nano")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--truncate-dim", type=int, default=1024)
    p.add_argument(
        "--voyage-holdout-dir",
        type=Path,
        default=Path("artifacts/voyage_nano_holdout"),
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    split_path = args.split_path or (args.data_dir / "synthetic" / "seed_split.json")
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    metrics = run_eval(
        data_dir=args.data_dir,
        split_path=split_path,
        artifacts_dir=args.artifacts_dir,
        fusion_mode=args.fusion,
        retrieval_mode=args.retrieval,
        rrf_k=args.rrf_k,
        logistic_C=args.logistic_C,
        max_features=args.max_features,
        ngram_range=(args.ngram_min, args.ngram_max),
        voyage_model=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        truncate_dim=args.truncate_dim,
        voyage_holdout_dir=args.voyage_holdout_dir,
        seed=args.seed,
    )
    print_metrics(metrics)
    print("\n=== Solo holdout (from caches, for attribution) ===")
    for k, v in metrics["solo_holdout"].items():
        print(f"{k}: {v:.4f}")

    out_path = args.artifacts_dir / "metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
