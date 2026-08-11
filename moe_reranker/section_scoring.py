"""Score sectioned seeker/candidate pairs under several aggregation shapes.

This is the MoE experiment's own copy of the scoring loop. It exists so the
veto-shaped aggregation modes can be evaluated **without editing
``baselines/voyage_nano_sectioned/``**, which the lookingFor-sectioning
experiment owns.

Isolation rule for this file: it may *import and call* shared baseline helpers,
but it must never modify them. Everything imported below is public API of those
modules (``load_pairs``, ``build_candidate_corpus``, ``build_sectioned_seekers``,
``VoyageNanoEncoder``, the metric functions) and is used read-only. The only
experiment-specific behaviour — which aggregation shapes exist — lives in
``moe_reranker.aggregation``.

Encoding is free when ``artifacts_dir`` already holds the cached embeddings:
``VoyageNanoEncoder`` keys its disk cache on a content hash that includes
``max_length``, so pass the value the cache was built with (4096 for
``artifacts/voyage_nano_sectioned_seeker_softmax_local``) or it will silently
re-encode for ~40 minutes on MPS.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from baselines.bert_frozen.text import candidate_to_text, seeker_to_text
from baselines.holdout import filter_to_holdout
from baselines.metrics import (
    pair_metrics,
    retrieval_metrics_from_ranks,
    slice_metrics,
)
from baselines.voyage_nano.encode import VoyageNanoEncoder, pick_device
from baselines.voyage_nano_sectioned.eval_seeker import (
    build_candidate_corpus,
    build_sectioned_seekers,
    load_pairs,
)
from moe_reranker.aggregation import aggregate_sections


def encode_sections(
    data_dir: Path,
    model_name: str,
    batch_size: int,
    max_length: int,
    truncate_dim: int | None,
    artifacts_dir: Path,
    *,
    holdout_only: bool = False,
    split_path: Path | None = None,
) -> dict[str, Any]:
    """Build the section-score matrices once, for every aggregation to reuse."""
    device = pick_device()
    print(f"device: {device}\nmodel:  {model_name}")

    positives, negatives = load_pairs(data_dir)
    if holdout_only:
        positives, negatives = filter_to_holdout(
            positives,
            negatives,
            split_path or data_dir / "synthetic" / "seed_split.json",
        )
        print(
            f"holdout-only: filtered to {len(positives)} positives, "
            f"{len(negatives)} negatives"
        )

    neg_cand_texts = [candidate_to_text(r["matchContactFile"]) for r in negatives]
    neg_seeker_texts_plain = [
        seeker_to_text(r["userContactFile"], r["searchQuery"]) for r in negatives
    ]

    corpus_ids, corpus_texts = build_candidate_corpus(positives, negatives)
    pos_offsets, pos_texts = build_sectioned_seekers(positives)
    neg_offsets, neg_texts = build_sectioned_seekers(negatives)
    print(f"candidate corpus size: {len(corpus_ids)}")
    print(f"seeker sections: {len(pos_texts)} (pos), {len(neg_texts)} (neg)")

    encoder = VoyageNanoEncoder(
        model_name=model_name,
        device=device,
        max_length=max_length,
        truncate_dim=truncate_dim,
        cache_dir=artifacts_dir,
    )
    pos_emb = encoder.encode(pos_texts, role="query", batch_size=batch_size)
    neg_emb = encoder.encode(neg_texts, role="query", batch_size=batch_size)
    corpus_emb = encoder.encode(corpus_texts, role="document", batch_size=batch_size)

    return {
        "device": device,
        "positives": positives,
        "negatives": negatives,
        "neg_cand_texts": neg_cand_texts,
        "neg_seeker_texts_plain": neg_seeker_texts_plain,
        "corpus_ids": corpus_ids,
        "id_to_idx": {c: i for i, c in enumerate(corpus_ids)},
        "pos_offsets": pos_offsets,
        "neg_offsets": neg_offsets,
        "pos_section_scores": pos_emb @ corpus_emb.T,
        "neg_section_scores": neg_emb @ corpus_emb.T,
    }


def score_one(
    encoded: dict[str, Any], *, agg: str, topk: int = 2, temperature: float = 0.05
) -> dict[str, Any]:
    """Aggregate the cached section-score matrices one way and build metrics.

    Note that the positive and negative matrices are aggregated in **separate
    calls** — which is exactly why every mode in ``moe_reranker.aggregation``
    must compute strictly within a record's own sections. A mode that derives a
    statistic from the matrix it is handed becomes label-dependent here.
    """
    positives, negatives = encoded["positives"], encoded["negatives"]
    id_to_idx, corpus_ids = encoded["id_to_idx"], encoded["corpus_ids"]

    pos_rec = aggregate_sections(
        encoded["pos_section_scores"], encoded["pos_offsets"],
        mode=agg, topk=topk, temperature=temperature,
    )
    neg_rec = aggregate_sections(
        encoded["neg_section_scores"], encoded["neg_offsets"],
        mode=agg, topk=topk, temperature=temperature,
    )

    pos_ids = [r["matchContactId"] for r in positives]
    neg_ids = [r["matchContactId"] for r in negatives]
    pos_scores = np.array([pos_rec[i, id_to_idx[t]] for i, t in enumerate(pos_ids)])
    neg_scores = np.array([neg_rec[i, id_to_idx[t]] for i, t in enumerate(neg_ids)])

    ranks: list[int] = []
    for i, target in enumerate(pos_ids):
        order = np.argsort(-pos_rec[i], kind="stable")
        ranks.append(int(np.where(order == id_to_idx[target])[0][0]) + 1)

    return {
        "agg": agg,
        "topk": topk,
        "temperature": temperature,
        "pair": pair_metrics(pos_scores, neg_scores),
        "retrieval": retrieval_metrics_from_ranks(ranks),
        "slices": slice_metrics(
            positives=positives,
            negatives=negatives,
            pos_scores=pos_scores,
            neg_scores=neg_scores,
            neg_seeker_texts=encoded["neg_seeker_texts_plain"],
            neg_cand_texts=encoded["neg_cand_texts"],
            query_embs=None,
            target_ids=pos_ids,
            candidate_ids=corpus_ids,
            candidate_embs=None,
            ranks=ranks,
        ),
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


def score_all_aggregations(
    *,
    data_dir: Path,
    model_name: str,
    batch_size: int,
    max_length: int,
    truncate_dim: int | None,
    artifacts_dir: Path,
    agg_configs: list[dict[str, Any]],
    holdout_only: bool = False,
    split_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Encode once, score every aggregation config. Returns {label: metrics}."""
    encoded = encode_sections(
        data_dir,
        model_name,
        batch_size,
        max_length,
        truncate_dim,
        artifacts_dir,
        holdout_only=holdout_only,
        split_path=split_path,
    )
    return {
        cfg.get("label", cfg["agg"]): score_one(
            encoded,
            agg=cfg["agg"],
            topk=cfg.get("topk", 2),
            temperature=cfg.get("temperature", 0.05),
        )
        for cfg in agg_configs
    }
