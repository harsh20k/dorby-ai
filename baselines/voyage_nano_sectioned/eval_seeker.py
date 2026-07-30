"""Eval for voyage-4-nano with the seeker's own lookingFor split into sections.

Mirror of eval.py in the other direction: here the *candidate* side stays a
single whole-profile embedding (today's baseline), and the *seeker* side
becomes N section-texts (profile + searchQuery unchanged, seeker's own
lookingFor swapped to one section at a time). A record's score is the max
cosine over its own seeker-section variants against its one candidate
embedding. See baselines/voyage_nano_sectioned/text.py::seeker_to_sectioned_texts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from baselines.bert_frozen.text import candidate_to_text, seeker_to_text
from baselines.holdout import filter_to_holdout
from baselines.metrics import (
    pair_metrics,
    print_metrics,
    retrieval_metrics_from_ranks,
    slice_metrics,
)
from baselines.voyage_nano.encode import VoyageNanoEncoder, pick_device
from baselines.voyage_nano_sectioned.aggregate import (
    AGG_FAMILY,
    AGG_MODES,
    aggregate_sections,
)
from baselines.voyage_nano_sectioned.text import seeker_to_sectioned_texts


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
    """Unique matchContactIds with first-seen matchContactFile text (unsectioned)."""
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


def build_sectioned_seekers(
    records: list[dict[str, Any]],
) -> tuple[list[int], list[str]]:
    """Flattened seeker section texts + per-record offsets (len(records) + 1)."""
    offsets: list[int] = [0]
    section_texts: list[str] = []
    for r in records:
        sections = seeker_to_sectioned_texts(r["userContactFile"], r["searchQuery"])
        section_texts.extend(sections)
        offsets.append(len(section_texts))
    return offsets, section_texts


def _load_and_encode(
    data_dir: Path,
    model_name: str,
    batch_size: int,
    max_length: int,
    truncate_dim: int | None,
    artifacts_dir: Path,
    *,
    holdout_only: bool = False,
    split_path: Path | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    """Load pairs, build the sectioned-seeker + candidate-corpus embeddings once.

    Shared by ``run_eval`` (single agg mode) and ``run_eval_multi_agg`` (several
    agg modes scored from one embedding pass) so re-running for a second or
    third aggregation mode never re-encodes anything.
    """
    device = pick_device()
    print(f"device: {device}")
    print(f"model:  {model_name}")

    positives, negatives = load_pairs(data_dir)
    if holdout_only:
        positives, negatives = filter_to_holdout(
            positives, negatives, split_path or data_dir / "synthetic" / "seed_split.json"
        )
        print(f"holdout-only: filtered to {len(positives)} positives, {len(negatives)} negatives")

    if user_id:
        positives = [r for r in positives if r["userContactId"] == user_id]
        negatives = [r for r in negatives if r["userContactId"] == user_id]
        print(f"user-id filter {user_id}: {len(positives)} positives, {len(negatives)} negatives")

    print(f"scoring {len(positives)} positives, {len(negatives)} negatives")

    neg_cand_texts = [candidate_to_text(r["matchContactFile"]) for r in negatives]
    # unaffected by sectioning; only used for the neg-hardness lexical-overlap slice
    neg_seeker_texts_plain = [
        seeker_to_text(r["userContactFile"], r["searchQuery"]) for r in negatives
    ]

    corpus_ids, corpus_texts = build_candidate_corpus(positives, negatives)
    id_to_idx = {cid: i for i, cid in enumerate(corpus_ids)}
    print(f"candidate corpus size: {len(corpus_ids)}")

    pos_offsets, pos_section_texts = build_sectioned_seekers(positives)
    neg_offsets, neg_section_texts = build_sectioned_seekers(negatives)
    print(
        f"seeker sections: {len(pos_section_texts)} (pos), {len(neg_section_texts)} (neg)"
    )

    encoder = VoyageNanoEncoder(
        model_name=model_name,
        device=device,
        max_length=max_length,
        truncate_dim=truncate_dim,
        cache_dir=artifacts_dir,
    )

    pos_section_emb = encoder.encode(
        pos_section_texts, role="query", batch_size=batch_size, cache_name=None
    )
    neg_section_emb = encoder.encode(
        neg_section_texts, role="query", batch_size=batch_size, cache_name=None
    )
    corpus_emb = encoder.encode(
        corpus_texts, role="document", batch_size=batch_size, cache_name=None
    )

    return {
        "device": device,
        "positives": positives,
        "negatives": negatives,
        "neg_cand_texts": neg_cand_texts,
        "neg_seeker_texts_plain": neg_seeker_texts_plain,
        "corpus_ids": corpus_ids,
        "id_to_idx": id_to_idx,
        "pos_offsets": pos_offsets,
        "neg_offsets": neg_offsets,
        "pos_section_scores": pos_section_emb @ corpus_emb.T,
        "neg_section_scores": neg_section_emb @ corpus_emb.T,
    }


def _score_with_agg(
    encoded: dict[str, Any],
    *,
    agg: str,
    topk: int,
    temperature: float,
) -> dict[str, Any]:
    """Aggregate the already-computed section-score matrices and build metrics."""
    positives = encoded["positives"]
    negatives = encoded["negatives"]
    id_to_idx = encoded["id_to_idx"]
    corpus_ids = encoded["corpus_ids"]

    pos_record_scores = aggregate_sections(
        encoded["pos_section_scores"],
        encoded["pos_offsets"],
        mode=agg,
        topk=topk,
        temperature=temperature,
    )
    neg_record_scores = aggregate_sections(
        encoded["neg_section_scores"],
        encoded["neg_offsets"],
        mode=agg,
        topk=topk,
        temperature=temperature,
    )

    pos_target_ids = [r["matchContactId"] for r in positives]
    neg_target_ids = [r["matchContactId"] for r in negatives]

    pos_scores = np.array(
        [pos_record_scores[i, id_to_idx[tid]] for i, tid in enumerate(pos_target_ids)]
    )
    neg_scores = np.array(
        [neg_record_scores[i, id_to_idx[tid]] for i, tid in enumerate(neg_target_ids)]
    )
    pair = pair_metrics(pos_scores, neg_scores)

    ranks: list[int] = []
    for i, target_id in enumerate(pos_target_ids):
        order = np.argsort(-pos_record_scores[i], kind="stable")
        rank = int(np.where(order == id_to_idx[target_id])[0][0]) + 1
        ranks.append(rank)
    retrieval = retrieval_metrics_from_ranks(ranks)

    slices = slice_metrics(
        positives=positives,
        negatives=negatives,
        pos_scores=pos_scores,
        neg_scores=neg_scores,
        neg_seeker_texts=encoded["neg_seeker_texts_plain"],
        neg_cand_texts=encoded["neg_cand_texts"],
        query_embs=None,
        target_ids=pos_target_ids,
        candidate_ids=corpus_ids,
        candidate_embs=None,
        ranks=ranks,
    )

    return {
        "agg": agg,
        "topk": topk,
        "temperature": temperature,
        "pair": pair,
        "retrieval": retrieval,
        "slices": slices,
        "per_pair_detail": {
            "positives": [
                {"matchContactId": r["matchContactId"], "score": float(s)}
                for r, s in zip(positives, pos_scores)
            ],
            "negatives": [
                {"matchContactId": r["matchContactId"], "score": float(s)}
                for r, s in zip(negatives, neg_scores)
            ],
        },
    }


def run_eval(
    data_dir: Path,
    model_name: str,
    batch_size: int,
    max_length: int,
    truncate_dim: int | None,
    artifacts_dir: Path,
    *,
    holdout_only: bool = False,
    split_path: Path | None = None,
    user_id: str | None = None,
    agg: str = "max",
    topk: int = 2,
    temperature: float = 0.05,
) -> dict[str, Any]:
    encoded = _load_and_encode(
        data_dir,
        model_name,
        batch_size,
        max_length,
        truncate_dim,
        artifacts_dir,
        holdout_only=holdout_only,
        split_path=split_path,
        user_id=user_id,
    )
    scored = _score_with_agg(encoded, agg=agg, topk=topk, temperature=temperature)
    return {
        "model_name": model_name,
        "device": str(encoded["device"]),
        "max_length": max_length,
        "truncate_dim": truncate_dim,
        "batch_size": batch_size,
        "user_id_filter": user_id,
        **scored,
    }


def run_eval_multi_agg(
    data_dir: Path,
    model_name: str,
    batch_size: int,
    max_length: int,
    truncate_dim: int | None,
    artifacts_dir: Path,
    agg_configs: list[dict[str, Any]],
    *,
    holdout_only: bool = False,
    split_path: Path | None = None,
    user_id: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Encode sections/candidates once, score several agg modes from that one pass.

    ``agg_configs`` is a list of dicts like
    ``{"label": "topk_mean", "agg": "topk_mean", "topk": 2, "temperature": 0.05}``.
    Returns ``{label: metrics_dict}``, each metrics_dict shaped like ``run_eval``'s
    return value (same "pair"/"retrieval"/"slices" keys).
    """
    encoded = _load_and_encode(
        data_dir,
        model_name,
        batch_size,
        max_length,
        truncate_dim,
        artifacts_dir,
        holdout_only=holdout_only,
        split_path=split_path,
        user_id=user_id,
    )
    results: dict[str, dict[str, Any]] = {}
    for cfg in agg_configs:
        label = cfg.get("label", cfg["agg"])
        scored = _score_with_agg(
            encoded,
            agg=cfg["agg"],
            topk=cfg.get("topk", 2),
            temperature=cfg.get("temperature", 0.05),
        )
        results[label] = {
            "model_name": model_name,
            "device": str(encoded["device"]),
            "max_length": max_length,
            "truncate_dim": truncate_dim,
            "batch_size": batch_size,
            "user_id_filter": user_id,
            **scored,
        }
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Voyage-4-nano, seeker's own lookingFor split into sections"
    )
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--model", type=str, default="voyageai/voyage-4-nano")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--truncate-dim", type=int, default=1024)
    p.add_argument(
        "--artifacts-dir", type=Path, default=Path("artifacts/voyage_nano_sectioned_seeker")
    )
    p.add_argument("--holdout-only", action="store_true")
    p.add_argument("--split-path", type=Path, default=None)
    p.add_argument("--user-id", type=str, default=None, help="Restrict to one userContactId")
    p.add_argument(
        "--agg",
        type=str,
        choices=list(AGG_MODES),
        default="max",
        help="Section-score aggregation mode (default 'max', backward compatible)",
    )
    p.add_argument(
        "--topk", type=int, default=2, help="k for --agg topk_mean (default 2)"
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.05,
        help="Temperature for --agg softmax (default 0.05)",
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
        holdout_only=args.holdout_only,
        split_path=args.split_path,
        user_id=args.user_id,
        agg=args.agg,
        topk=args.topk,
        temperature=args.temperature,
    )
    print_metrics(metrics)

    out_path = args.artifacts_dir / "metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
