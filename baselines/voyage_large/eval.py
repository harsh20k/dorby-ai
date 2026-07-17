"""Offline eval for Voyage-4-large API bi-encoder baseline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from baselines.bert_frozen.text import candidate_to_text, seeker_to_text
from baselines.voyage_large.encode import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_RPM_LIMIT,
    DEFAULT_TPM_LIMIT,
    VoyageLargeEncoder,
    cosine_scores,
    estimate_tokens,
    require_api_key,
)


def load_pairs(data_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pos_path = data_dir / "dataset_positive.json"
    neg_path = data_dir / "dataset_negative.json"
    if not pos_path.exists() or not neg_path.exists():
        raise FileNotFoundError(
            f"Expected {pos_path.name} and {neg_path.name} under {data_dir}"
        )
    positives = json.loads(pos_path.read_text())
    negatives = json.loads(neg_path.read_text())
    return positives, negatives


def build_candidate_corpus(
    positives: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """Unique matchContactIds with first-seen matchContactFile text."""
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


def pair_metrics(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
) -> dict[str, float]:
    y_true = np.concatenate(
        [np.ones(len(pos_scores), dtype=np.int32), np.zeros(len(neg_scores), dtype=np.int32)]
    )
    y_score = np.concatenate([pos_scores, neg_scores])
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "mean_cosine_positive": float(np.mean(pos_scores)),
        "mean_cosine_negative": float(np.mean(neg_scores)),
        "mean_cosine_gap": float(np.mean(pos_scores) - np.mean(neg_scores)),
        "num_positive": int(len(pos_scores)),
        "num_negative": int(len(neg_scores)),
    }


def retrieval_metrics(
    query_embs: np.ndarray,
    target_ids: list[str],
    candidate_ids: list[str],
    candidate_embs: np.ndarray,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    id_to_idx = {cid: i for i, cid in enumerate(candidate_ids)}
    ranks: list[int] = []
    recalls = {k: 0 for k in ks}

    for q_emb, target_id in zip(query_embs, target_ids):
        target_idx = id_to_idx[target_id]
        scores = candidate_embs @ q_emb  # both L2-normalized
        order = np.argsort(-scores, kind="stable")
        rank = int(np.where(order == target_idx)[0][0]) + 1  # 1-based
        ranks.append(rank)
        for k in ks:
            if rank <= k:
                recalls[k] += 1

    n = len(ranks)
    out: dict[str, float] = {
        "mrr": float(np.mean([1.0 / r for r in ranks])),
        "num_queries": float(n),
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
    }
    for k in ks:
        out[f"recall@{k}"] = float(recalls[k] / n) if n else 0.0
    # Alias for clarity in reports.
    out["top1"] = out.get("recall@1", 0.0)
    return out


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"warning: invalid {name}={raw!r}, using default {default}")
        return default


def run_eval(
    data_dir: Path,
    model_name: str,
    output_dimension: int,
    batch_size: int,
    artifacts_dir: Path,
    tpm_limit: int,
    rpm_limit: int,
) -> dict[str, Any]:
    print(f"model:             {model_name}")
    print(f"output_dimension:  {output_dimension}")
    print(f"batch_size:        {batch_size}")
    print(f"tpm_limit:         {tpm_limit}")
    print(f"rpm_limit:         {rpm_limit}")

    positives, negatives = load_pairs(data_dir)
    print(f"loaded {len(positives)} positives, {len(negatives)} negatives")

    pos_seeker_texts = [
        seeker_to_text(r["userContactFile"], r["searchQuery"]) for r in positives
    ]
    neg_seeker_texts = [
        seeker_to_text(r["userContactFile"], r["searchQuery"]) for r in negatives
    ]
    pos_cand_texts = [candidate_to_text(r["matchContactFile"]) for r in positives]
    neg_cand_texts = [candidate_to_text(r["matchContactFile"]) for r in negatives]

    corpus_ids, corpus_texts = build_candidate_corpus(positives, negatives)
    print(f"candidate corpus size: {len(corpus_ids)}")

    all_texts = (
        pos_seeker_texts
        + neg_seeker_texts
        + pos_cand_texts
        + neg_cand_texts
        + corpus_texts
    )
    est_tokens = estimate_tokens(all_texts)
    print(f"estimated tokens (naive, pre-dedupe/cache): ~{est_tokens:,}")
    if est_tokens > 2_000_000:
        print(
            "warning: estimated token budget is large; disk cache + dedupe will "
            "reduce actual API usage. Abort if this looks wrong."
        )

    encoder = VoyageLargeEncoder(
        model_name=model_name,
        output_dimension=output_dimension,
        truncation=True,
        cache_dir=artifacts_dir,
        tpm_limit=tpm_limit,
        rpm_limit=rpm_limit,
        batch_size=batch_size,
    )

    pos_seeker_emb = encoder.encode(
        pos_seeker_texts, input_type="query", batch_size=batch_size, label="pos_seeker"
    )
    neg_seeker_emb = encoder.encode(
        neg_seeker_texts, input_type="query", batch_size=batch_size, label="neg_seeker"
    )
    pos_cand_emb = encoder.encode(
        pos_cand_texts, input_type="document", batch_size=batch_size, label="pos_cand"
    )
    neg_cand_emb = encoder.encode(
        neg_cand_texts, input_type="document", batch_size=batch_size, label="neg_cand"
    )
    corpus_emb = encoder.encode(
        corpus_texts, input_type="document", batch_size=batch_size, label="corpus"
    )

    pos_scores = cosine_scores(pos_seeker_emb, pos_cand_emb)
    neg_scores = cosine_scores(neg_seeker_emb, neg_cand_emb)
    pair = pair_metrics(pos_scores, neg_scores)

    pos_target_ids = [r["matchContactId"] for r in positives]
    retrieval = retrieval_metrics(
        query_embs=pos_seeker_emb,
        target_ids=pos_target_ids,
        candidate_ids=corpus_ids,
        candidate_embs=corpus_emb,
    )

    usage = encoder.stats()
    usage_path = encoder.write_usage_meta()
    print(
        f"API usage: {usage['total_tokens_used']:,} tokens, "
        f"{usage['total_api_calls']} calls, "
        f"cache hits={usage['cache_hits']} misses={usage['cache_misses']}"
    )
    print(f"wrote {usage_path}")

    return {
        "model_name": model_name,
        "output_dimension": output_dimension,
        "batch_size": batch_size,
        "tpm_limit": tpm_limit,
        "rpm_limit": rpm_limit,
        "pair": pair,
        "retrieval": retrieval,
        "usage": usage,
    }


def print_metrics(metrics: dict[str, Any]) -> None:
    pair = metrics["pair"]
    ret = metrics["retrieval"]
    print("\n=== Pair metrics (200 labeled pairs) ===")
    print(f"ROC-AUC:            {pair['roc_auc']:.4f}")
    print(f"Average Precision:  {pair['average_precision']:.4f}")
    print(f"Mean cosine (pos):  {pair['mean_cosine_positive']:.4f}")
    print(f"Mean cosine (neg):  {pair['mean_cosine_negative']:.4f}")
    print(f"Mean cosine gap:    {pair['mean_cosine_gap']:.4f}")

    print("\n=== Retrieval metrics (100 positives vs corpus) ===")
    print(f"MRR:                {ret['mrr']:.4f}")
    print(f"Top-1 (= Recall@1): {ret['recall@1']:.4f}")
    print(f"Recall@1:           {ret['recall@1']:.4f}")
    print(f"Recall@5:           {ret['recall@5']:.4f}")
    print(f"Recall@10:          {ret['recall@10']:.4f}")
    print(f"Mean rank:          {ret['mean_rank']:.2f}")
    print(f"Median rank:        {ret['median_rank']:.1f}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Voyage-4-large API bi-encoder baseline eval")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--model", type=str, default="voyage-4-large")
    p.add_argument(
        "--output-dimension",
        type=int,
        default=1024,
        help="Matryoshka output dim (256/512/1024/2048). Default 1024 matches Boardy.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Texts per API request (max 1000; keep small for token budget).",
    )
    p.add_argument(
        "--tpm-limit",
        type=int,
        default=_env_int("VOYAGE_TPM_LIMIT", DEFAULT_TPM_LIMIT),
        help=f"Tokens/min throttle (default {DEFAULT_TPM_LIMIT}; env VOYAGE_TPM_LIMIT).",
    )
    p.add_argument(
        "--rpm-limit",
        type=int,
        default=_env_int("VOYAGE_RPM_LIMIT", DEFAULT_RPM_LIMIT),
        help=f"Requests/min throttle (default {DEFAULT_RPM_LIMIT}; env VOYAGE_RPM_LIMIT).",
    )
    p.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts/voyage_large"),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    require_api_key()  # fail fast before loading datasets
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    metrics = run_eval(
        data_dir=args.data_dir,
        model_name=args.model,
        output_dimension=args.output_dimension,
        batch_size=args.batch_size,
        artifacts_dir=args.artifacts_dir,
        tpm_limit=args.tpm_limit,
        rpm_limit=args.rpm_limit,
    )
    print_metrics(metrics)

    out_path = args.artifacts_dir / "metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
