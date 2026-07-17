"""Shared offline baseline metrics (pair / retrieval / slices).

Production online metrics (accept rate, intro success, etc.) are out of
scope for these offline baselines.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Pair scoring
# ---------------------------------------------------------------------------


def _score_percentiles(scores: np.ndarray) -> dict[str, float]:
    if len(scores) == 0:
        return {"p10": 0.0, "p50": 0.0, "p90": 0.0}
    p10, p50, p90 = np.percentile(scores, [10, 50, 90])
    return {"p10": float(p10), "p50": float(p50), "p90": float(p90)}


def _best_f1_threshold(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[float, float, float]:
    """Sweep PR-curve thresholds; return (best_f1, threshold, accuracy@threshold)."""
    if len(np.unique(y_true)) < 2 or len(y_score) == 0:
        return 0.0, 0.5, 0.0

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns len(thresholds) = len(precision) - 1
    best_f1, best_t, best_acc = -1.0, 0.5, 0.0
    if len(thresholds) == 0:
        pred = (y_score >= 0.5).astype(np.int32)
        return (
            float(f1_score(y_true, pred, zero_division=0)),
            0.5,
            float(accuracy_score(y_true, pred)),
        )

    for i, t in enumerate(thresholds):
        # precision[i], recall[i] correspond to threshold thresholds[i]
        p, r = float(precision[i]), float(recall[i])
        f1 = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
        if f1 > best_f1:
            pred = (y_score >= t).astype(np.int32)
            best_f1 = f1
            best_t = float(t)
            best_acc = float(accuracy_score(y_true, pred))

    return float(best_f1), float(best_t), float(best_acc)


def pair_metrics(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
) -> dict[str, Any]:
    y_true = np.concatenate(
        [np.ones(len(pos_scores), dtype=np.int32), np.zeros(len(neg_scores), dtype=np.int32)]
    )
    y_score = np.concatenate([pos_scores, neg_scores])
    best_f1, best_t, best_acc = _best_f1_threshold(y_true, y_score)
    acc_at_half = float(accuracy_score(y_true, (y_score >= 0.5).astype(np.int32)))

    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "best_f1": best_f1,
        "best_f1_threshold": best_t,
        "best_f1_accuracy": best_acc,
        "accuracy_at_0.5": acc_at_half,
        "mean_cosine_positive": float(np.mean(pos_scores)),
        "mean_cosine_negative": float(np.mean(neg_scores)),
        "mean_cosine_gap": float(np.mean(pos_scores) - np.mean(neg_scores)),
        "std_cosine_positive": float(np.std(pos_scores)),
        "std_cosine_negative": float(np.std(neg_scores)),
        "score_percentiles": {
            "positive": _score_percentiles(pos_scores),
            "negative": _score_percentiles(neg_scores),
        },
        "num_positive": int(len(pos_scores)),
        "num_negative": int(len(neg_scores)),
    }


# ---------------------------------------------------------------------------
# Retrieval / ranking (binary relevance; one labeled positive per query)
# ---------------------------------------------------------------------------


def _dcg_at_k(rank: int, k: int) -> float:
    """Binary DCG@K for a single relevant item at 1-based `rank`."""
    if rank > k:
        return 0.0
    return 1.0 / np.log2(rank + 1)


def retrieval_metrics(
    query_embs: np.ndarray,
    target_ids: list[str],
    candidate_ids: list[str],
    candidate_embs: np.ndarray,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Ranking metrics with binary labels (1 if labeled positive match else 0).

    Soft / graded relevance would unlock true graded NDCG later; here NDCG is
    binary. With one relevant item per query, MAP equals mean AP = MRR.
    Precision@K = 1/K if rank<=K else 0 (mean over queries).
    """
    id_to_idx = {cid: i for i, cid in enumerate(candidate_ids)}
    ranks: list[int] = []
    recalls = {k: 0 for k in ks}
    ndcgs = {k: [] for k in ks}
    precisions = {k: [] for k in ks}
    aps: list[float] = []

    # Ideal DCG for one relevant item at rank 1
    idcg = 1.0 / np.log2(2)

    for q_emb, target_id in zip(query_embs, target_ids):
        target_idx = id_to_idx[target_id]
        scores = candidate_embs @ q_emb  # both L2-normalized
        order = np.argsort(-scores, kind="stable")
        rank = int(np.where(order == target_idx)[0][0]) + 1  # 1-based
        ranks.append(rank)
        aps.append(1.0 / rank)  # single relevant → AP = 1/rank
        for k in ks:
            if rank <= k:
                recalls[k] += 1
                precisions[k].append(1.0 / k)
            else:
                precisions[k].append(0.0)
            ndcgs[k].append(_dcg_at_k(rank, k) / idcg)

    n = len(ranks)
    out: dict[str, float] = {
        "mrr": float(np.mean([1.0 / r for r in ranks])) if n else 0.0,
        "map": float(np.mean(aps)) if n else 0.0,
        "num_queries": float(n),
        "mean_rank": float(np.mean(ranks)) if n else 0.0,
        "median_rank": float(np.median(ranks)) if n else 0.0,
    }
    for k in ks:
        out[f"recall@{k}"] = float(recalls[k] / n) if n else 0.0
        out[f"ndcg@{k}"] = float(np.mean(ndcgs[k])) if n else 0.0
        out[f"precision@{k}"] = float(np.mean(precisions[k])) if n else 0.0
    out["top1"] = out.get("recall@1", 0.0)
    return out


def ranks_for_queries(
    query_embs: np.ndarray,
    target_ids: list[str],
    candidate_ids: list[str],
    candidate_embs: np.ndarray,
) -> list[int]:
    """1-based ranks of labeled targets (helper for slice metrics)."""
    id_to_idx = {cid: i for i, cid in enumerate(candidate_ids)}
    ranks: list[int] = []
    for q_emb, target_id in zip(query_embs, target_ids):
        target_idx = id_to_idx[target_id]
        scores = candidate_embs @ q_emb
        order = np.argsort(-scores, kind="stable")
        ranks.append(int(np.where(order == target_idx)[0][0]) + 1)
    return ranks


# ---------------------------------------------------------------------------
# Intent slices
# ---------------------------------------------------------------------------

_INTENT_RULES: list[tuple[str, tuple[str, ...]]] = [
    (
        "fundraise",
        (
            "fundrais",
            "investor",
            "investors",
            "venture capital",
            " vc",
            "seed round",
            "pre-seed",
            "preseed",
            "series a",
            "series b",
            "capital raise",
            "deal flow",
            "limited partner",
            " lp ",
            "angel",
        ),
    ),
    (
        "hiring",
        (
            "hiring",
            "talent",
            "recruit",
            "job seek",
            "jobs",
            "candidate",
            "headhunt",
            "executive role",
            "general manager",
            "looking for people",
            "founding engineer",
            "ai engineer",
            "software engineer",
        ),
    ),
    (
        "partnerships",
        (
            "partner",
            "gtm",
            "go-to-market",
            "go to market",
            "channel partner",
            "alliance",
            "co-sell",
            "distribution partner",
        ),
    ),
    (
        "customers",
        (
            "customer",
            "brand-side",
            "brand side",
            "buyer",
            "buyers",
            "operations leader",
            "ops ",
            "dtc",
            "ecommerce",
            "e-commerce",
            "fulfillment",
            "procurement",
            "enterprise intro",
            "enterprise prospect",
        ),
    ),
]


def classify_intent(search_query: str | None, looking_for: str | None = None) -> str:
    """Coarse intent bucket from searchQuery (+ lookingFor) via keyword heuristics."""
    text = f"{search_query or ''} {looking_for or ''}".lower()
    for name, kws in _INTENT_RULES:
        if any(k in text for k in kws):
            return name
    return "other"


def _record_intent(record: dict[str, Any]) -> str:
    looking = None
    ucf = record.get("userContactFile") or {}
    if isinstance(ucf, dict):
        looking = ucf.get("lookingFor")
    return classify_intent(record.get("searchQuery"), looking)


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def intent_slice_metrics(
    positives: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
    query_embs: np.ndarray,
    target_ids: list[str],
    candidate_ids: list[str],
    candidate_embs: np.ndarray,
    min_n: int = 5,
) -> dict[str, Any]:
    """Per-intent pair AUC + retrieval MRR / Recall@10. Skip/mark slices with n < min_n."""
    pos_intents = [_record_intent(r) for r in positives]
    neg_intents = [_record_intent(r) for r in negatives]
    ranks = ranks_for_queries(query_embs, target_ids, candidate_ids, candidate_embs)

    buckets = sorted(set(pos_intents) | set(neg_intents))
    out: dict[str, Any] = {}

    for bucket in buckets:
        pos_idx = [i for i, b in enumerate(pos_intents) if b == bucket]
        neg_idx = [i for i, b in enumerate(neg_intents) if b == bucket]
        n_pair = len(pos_idx) + len(neg_idx)
        n_queries = len(pos_idx)

        slice_out: dict[str, Any] = {
            "n_pairs": n_pair,
            "n_queries": n_queries,
            "low_n": n_pair < min_n and n_queries < min_n,
        }

        if len(pos_idx) >= 1 and len(neg_idx) >= 1 and n_pair >= min_n:
            y_true = np.concatenate(
                [np.ones(len(pos_idx), dtype=np.int32), np.zeros(len(neg_idx), dtype=np.int32)]
            )
            y_score = np.concatenate([pos_scores[pos_idx], neg_scores[neg_idx]])
            auc = _safe_auc(y_true, y_score)
            slice_out["pair_auc"] = auc
        else:
            slice_out["pair_auc"] = None
            if n_pair < min_n:
                slice_out["low_n"] = True

        if n_queries >= min_n:
            rs = [ranks[i] for i in pos_idx]
            slice_out["mrr"] = float(np.mean([1.0 / r for r in rs]))
            slice_out["recall@10"] = float(np.mean([1.0 if r <= 10 else 0.0 for r in rs]))
        else:
            slice_out["mrr"] = None
            slice_out["recall@10"] = None
            if n_queries < min_n:
                slice_out["low_n"] = True

        out[bucket] = slice_out

    return out


# ---------------------------------------------------------------------------
# Hard-neg vs easy-neg (lexical overlap on negative pairs)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def token_jaccard(a: str, b: str) -> float:
    """Jaccard overlap on alphanumeric tokens (lowercase)."""
    ta = set(_TOKEN_RE.findall(a.lower()))
    tb = set(_TOKEN_RE.findall(b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def neg_hardness_slice_metrics(
    neg_scores: np.ndarray,
    neg_seeker_texts: list[str],
    neg_cand_texts: list[str],
    pos_scores: np.ndarray | None = None,
    easy_quantile: float = 0.25,
    hard_quantile: float = 0.50,
) -> dict[str, Any]:
    """Split negatives by seeker↔match token Jaccard; report pair AUC / gaps.

    Rule (documented):
      - Compute Jaccard(token(seeker_text), token(match_text)) per negative pair.
      - **easy-neg**: overlap <= `easy_quantile` of the negative-pair overlap
        distribution (near-zero / low topical lexical overlap).
      - **hard-neg**: overlap >= `hard_quantile` (mid–high lexical overlap;
        superficially similar text that is still labeled negative).

    Pair AUC for a hardness bucket uses all positives + that bucket's negatives
    (positives are the same reference set). Also reports mean cosine gap
    (mean_pos - mean_neg_bucket).
    """
    n = len(neg_scores)
    overlaps = np.array(
        [token_jaccard(s, c) for s, c in zip(neg_seeker_texts, neg_cand_texts)],
        dtype=np.float64,
    )
    if n == 0:
        return {
            "rule": {
                "easy": f"jaccard <= p{int(easy_quantile * 100)} of neg overlaps",
                "hard": f"jaccard >= p{int(hard_quantile * 100)} of neg overlaps",
            },
            "easy": {},
            "hard": {},
        }

    easy_cut = float(np.quantile(overlaps, easy_quantile))
    hard_cut = float(np.quantile(overlaps, hard_quantile))
    easy_mask = overlaps <= easy_cut
    hard_mask = overlaps >= hard_cut

    mean_pos = float(np.mean(pos_scores)) if pos_scores is not None and len(pos_scores) else None

    def _bucket(mask: np.ndarray, label: str) -> dict[str, Any]:
        idx = np.where(mask)[0]
        bucket_scores = neg_scores[idx]
        out: dict[str, Any] = {
            "n_negatives": int(len(idx)),
            "mean_overlap": float(np.mean(overlaps[idx])) if len(idx) else None,
            "overlap_cutoff": easy_cut if label == "easy" else hard_cut,
            "low_n": len(idx) < 5,
        }
        if mean_pos is not None and len(idx):
            out["mean_cosine_negative"] = float(np.mean(bucket_scores))
            out["mean_cosine_gap"] = float(mean_pos - np.mean(bucket_scores))
        else:
            out["mean_cosine_negative"] = None
            out["mean_cosine_gap"] = None

        if (
            pos_scores is not None
            and len(pos_scores) >= 1
            and len(idx) >= 5
            and len(np.unique(np.concatenate([np.ones(len(pos_scores)), np.zeros(len(idx))])))
            >= 2
        ):
            y_true = np.concatenate(
                [np.ones(len(pos_scores), dtype=np.int32), np.zeros(len(idx), dtype=np.int32)]
            )
            y_score = np.concatenate([pos_scores, bucket_scores])
            out["pair_auc"] = _safe_auc(y_true, y_score)
        else:
            out["pair_auc"] = None
        return out

    return {
        "rule": {
            "overlap": "token Jaccard(seeker_text, match_text) on negative pairs",
            "easy": f"jaccard <= p{int(easy_quantile * 100)} (cutoff={easy_cut:.4f})",
            "hard": f"jaccard >= p{int(hard_quantile * 100)} (cutoff={hard_cut:.4f})",
        },
        "overlap_stats": {
            "mean": float(np.mean(overlaps)),
            "p25": float(np.quantile(overlaps, 0.25)),
            "p50": float(np.quantile(overlaps, 0.50)),
            "p75": float(np.quantile(overlaps, 0.75)),
        },
        "easy": _bucket(easy_mask, "easy"),
        "hard": _bucket(hard_mask, "hard"),
    }


def slice_metrics(
    positives: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
    neg_seeker_texts: list[str],
    neg_cand_texts: list[str],
    query_embs: np.ndarray,
    target_ids: list[str],
    candidate_ids: list[str],
    candidate_embs: np.ndarray,
) -> dict[str, Any]:
    return {
        "intent": intent_slice_metrics(
            positives,
            negatives,
            pos_scores,
            neg_scores,
            query_embs,
            target_ids,
            candidate_ids,
            candidate_embs,
        ),
        "neg_hardness": neg_hardness_slice_metrics(
            neg_scores,
            neg_seeker_texts,
            neg_cand_texts,
            pos_scores=pos_scores,
        ),
    }


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------


def print_metrics(metrics: dict[str, Any]) -> None:
    """Print Pair / Retrieval / Slices sections. Production metrics are OOS offline."""
    pair = metrics["pair"]
    ret = metrics["retrieval"]
    slices = metrics.get("slices") or {}

    print("\n=== Pair metrics ===")
    print(f"ROC-AUC:              {pair['roc_auc']:.4f}")
    print(f"Average Precision:    {pair['average_precision']:.4f}")
    print(
        f"Best-F1:              {pair['best_f1']:.4f} "
        f"@ threshold={pair['best_f1_threshold']:.4f} "
        f"(acc={pair['best_f1_accuracy']:.4f})"
    )
    print(f"Accuracy @ 0.5:       {pair['accuracy_at_0.5']:.4f}")
    print(
        f"Mean cosine (pos/neg/gap): "
        f"{pair['mean_cosine_positive']:.4f} / "
        f"{pair['mean_cosine_negative']:.4f} / "
        f"{pair['mean_cosine_gap']:.4f}"
    )
    print(
        f"Std cosine (pos/neg): "
        f"{pair['std_cosine_positive']:.4f} / {pair['std_cosine_negative']:.4f}"
    )
    sp = pair.get("score_percentiles") or {}
    pos_p = sp.get("positive") or {}
    neg_p = sp.get("negative") or {}
    if pos_p and neg_p:
        print(
            f"Cosine p10/p50/p90 pos: "
            f"{pos_p['p10']:.4f} / {pos_p['p50']:.4f} / {pos_p['p90']:.4f}"
        )
        print(
            f"Cosine p10/p50/p90 neg: "
            f"{neg_p['p10']:.4f} / {neg_p['p50']:.4f} / {neg_p['p90']:.4f}"
        )

    print("\n=== Retrieval metrics ===")
    print(
        f"MRR / mean_rank / median_rank: "
        f"{ret['mrr']:.4f} / {ret['mean_rank']:.2f} / {ret['median_rank']:.1f}"
    )
    print(f"MAP (≈ MRR w/ 1 relevant): {ret['map']:.4f}")
    print(
        f"NDCG@1/5/10 (binary):  "
        f"{ret['ndcg@1']:.4f} / {ret['ndcg@5']:.4f} / {ret['ndcg@10']:.4f}"
    )
    print("  (soft labels would unlock graded NDCG later)")
    print(
        f"Precision@1/5/10:     "
        f"{ret['precision@1']:.4f} / {ret['precision@5']:.4f} / {ret['precision@10']:.4f}"
    )
    print("  (1 labeled good → P@K = 1/K if rank≤K else 0; mean over queries)")
    print(
        f"Recall@1/5/10:        "
        f"{ret['recall@1']:.4f} / {ret['recall@5']:.4f} / {ret['recall@10']:.4f}"
    )

    intent = slices.get("intent") or {}
    hardness = slices.get("neg_hardness") or {}
    if intent or hardness:
        print("\n=== Slice metrics ===")
        if intent:
            print("By intent:")
            for name, s in sorted(intent.items()):
                flag = " [low-n]" if s.get("low_n") else ""
                auc = s.get("pair_auc")
                mrr = s.get("mrr")
                r10 = s.get("recall@10")
                auc_s = f"{auc:.4f}" if auc is not None else "n/a"
                mrr_s = f"{mrr:.4f}" if mrr is not None else "n/a"
                r10_s = f"{r10:.4f}" if r10 is not None else "n/a"
                print(
                    f"  {name:14s}  n_pairs={s.get('n_pairs', 0):3d}  "
                    f"n_q={s.get('n_queries', 0):3d}  "
                    f"AUC={auc_s}  MRR={mrr_s}  R@10={r10_s}{flag}"
                )
        if hardness:
            rule = hardness.get("rule") or {}
            print("Neg hardness (lexical Jaccard on neg pairs):")
            if rule:
                print(f"  easy: {rule.get('easy', '')}")
                print(f"  hard: {rule.get('hard', '')}")
            for label in ("easy", "hard"):
                b = hardness.get(label) or {}
                if not b:
                    continue
                flag = " [low-n]" if b.get("low_n") else ""
                auc = b.get("pair_auc")
                gap = b.get("mean_cosine_gap")
                auc_s = f"{auc:.4f}" if auc is not None else "n/a"
                gap_s = f"{gap:.4f}" if gap is not None else "n/a"
                print(
                    f"  {label:14s}  n_neg={b.get('n_negatives', 0):3d}  "
                    f"AUC={auc_s}  gap={gap_s}{flag}"
                )

    print(
        "\nNote: production metrics (accept rate, intro success, etc.) "
        "are out of scope for offline baselines."
    )
