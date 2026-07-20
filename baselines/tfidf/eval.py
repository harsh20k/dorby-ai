"""Offline eval for a TF-IDF lexical-similarity baseline.

Same seeker/candidate text serialization and metrics as the neural
baselines (bert_frozen, voyage_nano, voyage_large) — the only difference is
the encoder: plain TF-IDF cosine similarity, no neural model, no GPU. Exists
to establish a lexical floor: if this scores anywhere close to the neural
baselines, the embedding models aren't adding much beyond keyword overlap
for this task.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from baselines.bert_frozen.text import candidate_to_text, seeker_to_text
from baselines.holdout import filter_to_holdout
from baselines.metrics import pair_metrics, print_metrics, retrieval_metrics, slice_metrics
from baselines.tfidf.encode import TfidfEncoder, cosine_scores, pick_device


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


def run_eval(
    data_dir: Path,
    max_features: int,
    ngram_range: tuple[int, int],
    artifacts_dir: Path,
    *,
    holdout_only: bool = False,
    split_path: Path | None = None,
) -> dict[str, Any]:
    device = pick_device()
    print(f"device: {device}")
    print(f"model:  tfidf(max_features={max_features}, ngram_range={ngram_range})")

    positives, negatives = load_pairs(data_dir)
    if holdout_only:
        positives, negatives = filter_to_holdout(
            positives, negatives, split_path or data_dir / "synthetic" / "seed_split.json"
        )
        print(f"holdout-only: filtered to {len(positives)} positives, {len(negatives)} negatives")
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

    encoder = TfidfEncoder(
        max_features=max_features,
        ngram_range=ngram_range,
        cache_dir=artifacts_dir,
    )
    # Fit on every text that will be encoded, seeker + candidate pooled, so
    # cosine similarity between the two sides lives in a shared vocab/IDF
    # space. Candidate corpus already covers pos_cand/neg_cand (same texts
    # by matchContactId), only the seeker texts are additional.
    encoder.fit(pos_seeker_texts + neg_seeker_texts + corpus_texts)

    pos_seeker_emb = encoder.encode(pos_seeker_texts, cache_name="pos_seeker")
    neg_seeker_emb = encoder.encode(neg_seeker_texts, cache_name="neg_seeker")
    pos_cand_emb = encoder.encode(pos_cand_texts, cache_name="pos_cand")
    neg_cand_emb = encoder.encode(neg_cand_texts, cache_name="neg_cand")
    corpus_emb = encoder.encode(corpus_texts, cache_name="corpus")

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
        "model_name": f"tfidf(max_features={max_features},ngram_range={ngram_range})",
        "device": device,
        "max_features": max_features,
        "ngram_range": list(ngram_range),
        "pair": pair,
        "retrieval": retrieval,
        "slices": slices,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TF-IDF lexical-similarity baseline eval")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--max-features", type=int, default=20000)
    p.add_argument("--ngram-min", type=int, default=1)
    p.add_argument("--ngram-max", type=int, default=2)
    p.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts/tfidf"),
    )
    p.add_argument(
        "--holdout-only",
        action="store_true",
        help="Restrict eval to the frozen real holdout (data/synthetic/seed_split.json's "
        "eval_pair_ids) instead of the full canonical dataset. Pass a distinct "
        "--artifacts-dir so this doesn't overwrite the full-dataset metrics.json.",
    )
    p.add_argument("--split-path", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    metrics = run_eval(
        data_dir=args.data_dir,
        max_features=args.max_features,
        ngram_range=(args.ngram_min, args.ngram_max),
        artifacts_dir=args.artifacts_dir,
        holdout_only=args.holdout_only,
        split_path=args.split_path,
    )
    print_metrics(metrics)

    out_path = args.artifacts_dir / "metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
