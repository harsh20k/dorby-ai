"""Pick one real pair that shows the calibrated alpha's effect concretely.

Operates on ``query_weighted.eval.Encoded`` (from ``encode_everything``, called
unmodified) — reuses its already-computed ``concat_baseline``/``profile_only``/
``query_only`` vectors and candidate lookup rather than re-encoding anything.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from eval_real_full.baseline_eval import build_candidate_corpus
from query_weighted.eval import Encoded, combine


def _rank_of_target(seeker_vec: np.ndarray, target_id: str, corpus_ids: list[str], corpus_embs: np.ndarray) -> tuple[int, np.ndarray]:
    scores = corpus_embs @ seeker_vec
    order = np.argsort(-scores, kind="stable")
    target_idx = corpus_ids.index(target_id)
    rank = int(np.where(order == target_idx)[0][0]) + 1
    return rank, scores


def _top_k(scores: np.ndarray, corpus_ids: list[str], k: int, target_id: str) -> list[dict[str, Any]]:
    order = np.argsort(-scores, kind="stable")[:k]
    return [
        {
            "candidate_id": corpus_ids[i],
            "score": float(scores[i]),
            "is_target": corpus_ids[i] == target_id,
        }
        for i in order
    ]


def pick_worked_example(enc: Encoded, alpha: float, k: int = 5) -> dict[str, Any]:
    """The positive pair with the largest rank improvement, concat_baseline -> alpha blend.

    Ties broken by the worst concat_baseline rank, so the example is the most
    dramatic honestly-found case rather than a cherry-picked easy win.
    """
    corpus_ids, corpus_texts = build_candidate_corpus(enc.positives, enc.negatives)
    corpus_embs = np.stack([enc.doc_by_text[t] for t in corpus_texts])

    alpha_seeker = combine(enc.seeker["query_only"], enc.seeker["profile_only"], alpha)

    candidates: list[dict[str, Any]] = []
    for k_idx, i in enumerate(enc.pos_index):
        record = enc.positives[k_idx]
        target_id = record["matchContactId"]

        rank_concat, scores_concat = _rank_of_target(
            enc.seeker["concat_baseline"][i], target_id, corpus_ids, corpus_embs
        )
        rank_alpha, scores_alpha = _rank_of_target(
            alpha_seeker[i], target_id, corpus_ids, corpus_embs
        )

        candidates.append(
            {
                "pair_id": enc.pair_ids[i],
                "seeker_id": record["userContactId"],
                "target_id": target_id,
                "search_query": record["searchQuery"],
                "rank_concat_baseline": rank_concat,
                "rank_alpha": rank_alpha,
                "improvement": rank_concat - rank_alpha,
                "top_k_concat_baseline": _top_k(scores_concat, corpus_ids, k, target_id),
                "top_k_alpha": _top_k(scores_alpha, corpus_ids, k, target_id),
            }
        )

    best = max(candidates, key=lambda c: (c["improvement"], c["rank_concat_baseline"]))
    return best
