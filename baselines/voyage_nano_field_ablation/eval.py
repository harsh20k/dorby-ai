"""Offline eval for local Voyage-4-nano with exactly one candidate-side
profile field ablated — the downstream-metric analog of the earlier
embedding-distance ablation idea: instead of measuring how much removing a
field moves the embedding vector, measure how much it moves pair AUC /
retrieval quality, the thing the project actually optimizes for.

Seeker-side text is unchanged (full profile + searchQuery), so seeker
embeddings are identical across every field's ablation run and across the
unablated baseline. This script copies the seeker-side cache files
(`emb_pos_seeker.*`, `emb_neg_seeker.*`) from `--baseline-cache-dir` into
`--artifacts-dir` before encoding, so the (expensive, cold ~40min) seeker
MPS encode only ever runs once — every ablation run after the first only
pays for its own candidate-side + corpus encode, which is cheap since this
defaults to holdout-only (69 pairs total).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from baselines.bert_frozen.text import seeker_to_text
from baselines.holdout import filter_to_holdout
from baselines.metrics import pair_metrics, print_metrics, retrieval_metrics, slice_metrics
from baselines.text_field_ablation import PROFILE_FIELDS, candidate_to_text_ablate_field
from baselines.voyage_nano.encode import VoyageNanoEncoder, cosine_scores, pick_device
from baselines.voyage_nano.eval import load_pairs


def build_candidate_corpus_ablated(
    positives: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
    ablate_field: str,
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
        texts.append(candidate_to_text_ablate_field(record["matchContactFile"], ablate_field))
    return ids, texts


def seed_seeker_cache(baseline_cache_dir: Path, artifacts_dir: Path) -> None:
    """Copy the two seeker-side cache file pairs so this run's encoder hits
    cache immediately for pos_seeker/neg_seeker instead of re-running MPS."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for cache_name in ("pos_seeker", "neg_seeker"):
        for suffix in ("npy", "json"):
            src = baseline_cache_dir / f"emb_{cache_name}.{suffix}"
            if src.exists():
                shutil.copy2(src, artifacts_dir / f"emb_{cache_name}.{suffix}")


def run_eval(
    data_dir: Path,
    model_name: str,
    batch_size: int,
    max_length: int,
    truncate_dim: int | None,
    artifacts_dir: Path,
    ablate_field: str,
    *,
    holdout_only: bool = True,
    split_path: Path | None = None,
    baseline_cache_dir: Path | None = None,
) -> dict[str, Any]:
    device = pick_device()
    print(f"device: {device}")
    print(f"model:  {model_name}")
    print(f"ablating candidate field: {ablate_field}")

    positives, negatives = load_pairs(data_dir)
    if holdout_only:
        positives, negatives = filter_to_holdout(
            positives, negatives, split_path or data_dir / "synthetic" / "seed_split.json"
        )
        print(f"holdout-only: filtered to {len(positives)} positives, {len(negatives)} negatives")
    print(f"loaded {len(positives)} positives, {len(negatives)} negatives")

    if baseline_cache_dir is not None:
        seed_seeker_cache(baseline_cache_dir, artifacts_dir)

    pos_seeker_texts = [seeker_to_text(r["userContactFile"], r["searchQuery"]) for r in positives]
    neg_seeker_texts = [seeker_to_text(r["userContactFile"], r["searchQuery"]) for r in negatives]
    pos_cand_texts = [
        candidate_to_text_ablate_field(r["matchContactFile"], ablate_field) for r in positives
    ]
    neg_cand_texts = [
        candidate_to_text_ablate_field(r["matchContactFile"], ablate_field) for r in negatives
    ]

    corpus_ids, corpus_texts = build_candidate_corpus_ablated(positives, negatives, ablate_field)
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
        "ablated_field": ablate_field,
        "holdout_only": holdout_only,
        "pair": pair,
        "retrieval": retrieval,
        "slices": slices,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--model", type=str, default="voyageai/voyage-4-nano")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--truncate-dim", type=int, default=1024)
    p.add_argument("--ablate-field", type=str, required=True, choices=PROFILE_FIELDS)
    p.add_argument("--artifacts-dir", type=Path, required=True)
    p.add_argument(
        "--baseline-cache-dir",
        type=Path,
        default=None,
        help="artifacts dir of a prior unablated voyage_nano --holdout-only run, "
        "used only to seed this run's seeker-side cache (identical texts) so the "
        "cold MPS encode of the seeker side only happens once across all ablations",
    )
    p.add_argument("--holdout-only", action="store_true", default=True)
    p.add_argument("--full-dataset", dest="holdout_only", action="store_false")
    p.add_argument("--split-path", type=Path, default=None)
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
        ablate_field=args.ablate_field,
        holdout_only=args.holdout_only,
        split_path=args.split_path,
        baseline_cache_dir=args.baseline_cache_dir,
    )
    print_metrics(metrics)

    out_path = args.artifacts_dir / "metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
