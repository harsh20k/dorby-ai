"""Offline eval for Voyage-4-nano — seeker profile only (no searchQuery)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from baselines.metrics import pair_metrics, print_metrics, retrieval_metrics, slice_metrics
from baselines.text_no_query import candidate_to_text, seeker_to_text
from baselines.voyage_nano.encode import VoyageNanoEncoder, cosine_scores, pick_device
from baselines.voyage_nano.eval import build_candidate_corpus, load_pairs

SEEKER_TEXT_MODE = "profile_only_no_query"


def run_eval(
    data_dir: Path,
    model_name: str,
    batch_size: int,
    max_length: int,
    truncate_dim: int | None,
    artifacts_dir: Path,
) -> dict[str, Any]:
    device = pick_device()
    print(f"device: {device}")
    print(f"model:  {model_name}")
    print(f"seeker: {SEEKER_TEXT_MODE}")

    positives, negatives = load_pairs(data_dir)
    print(f"loaded {len(positives)} positives, {len(negatives)} negatives")

    pos_seeker_texts = [seeker_to_text(r["userContactFile"]) for r in positives]
    neg_seeker_texts = [seeker_to_text(r["userContactFile"]) for r in negatives]
    pos_cand_texts = [candidate_to_text(r["matchContactFile"]) for r in positives]
    neg_cand_texts = [candidate_to_text(r["matchContactFile"]) for r in negatives]

    corpus_ids, corpus_texts = build_candidate_corpus(positives, negatives)
    print(f"candidate corpus size: {len(corpus_ids)}")

    encoder = VoyageNanoEncoder(
        model_name=model_name,
        device=device,
        max_length=max_length,
        truncate_dim=truncate_dim,
        cache_dir=artifacts_dir,
    )

    pos_seeker_emb = encoder.encode(
        pos_seeker_texts, role="query", batch_size=batch_size, cache_name="pos_seeker"
    )
    neg_seeker_emb = encoder.encode(
        neg_seeker_texts, role="query", batch_size=batch_size, cache_name="neg_seeker"
    )
    pos_cand_emb = encoder.encode(
        pos_cand_texts, role="document", batch_size=batch_size, cache_name="pos_cand"
    )
    neg_cand_emb = encoder.encode(
        neg_cand_texts, role="document", batch_size=batch_size, cache_name="neg_cand"
    )
    corpus_emb = encoder.encode(
        corpus_texts, role="document", batch_size=batch_size, cache_name="corpus"
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
    slices = slice_metrics(
        positives=positives,
        negatives=negatives,
        pos_scores=pos_scores,
        neg_scores=neg_scores,
        neg_seeker_texts=neg_seeker_texts,
        neg_cand_texts=neg_cand_texts,
        query_embs=pos_seeker_emb,
        target_ids=pos_target_ids,
        candidate_ids=corpus_ids,
        candidate_embs=corpus_emb,
    )

    return {
        "model_name": model_name,
        "device": str(device),
        "max_length": max_length,
        "truncate_dim": truncate_dim,
        "batch_size": batch_size,
        "seeker_text": SEEKER_TEXT_MODE,
        "pair": pair,
        "retrieval": retrieval,
        "slices": slices,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Voyage-4-nano local bi-encoder baseline eval (no searchQuery on seeker)"
    )
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--model", type=str, default="voyageai/voyage-4-nano")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument(
        "--max-length",
        type=int,
        default=8192,
        help="Token cap (model supports 32k; 8192 default for MPS memory).",
    )
    p.add_argument(
        "--truncate-dim",
        type=int,
        default=1024,
        help="Matryoshka output dim (256/512/1024/2048). Default 1024 matches Boardy large.",
    )
    p.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts/voyage_nano_no_query"),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    metrics = run_eval(
        data_dir=args.data_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        truncate_dim=args.truncate_dim,
        artifacts_dir=args.artifacts_dir,
    )
    print_metrics(metrics)

    out_path = args.artifacts_dir / "metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
