"""Metrics specific to a judge that emits a decision, not just a score.

The embedding baselines only produce a cosine, so ``baselines/metrics.py``
has no notion of a model *committing* to an answer. An LLM judge does, and
that decision is the thing the experiment actually asked about — so it gets
measured directly here, alongside the shared ``pair_metrics`` (which stays the
single source of truth for AUC/AP and is used unmodified) so this row is
comparable to every other baseline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

# Intent bucketing is shared with the other baselines on purpose: the bucket
# definition must not fork, or per-intent numbers stop being comparable.
from baselines.metrics import _record_intent, _safe_auc


def decision_metrics(
    pos_verdicts: list[dict[str, Any]],
    neg_verdicts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score the model's literal yes/no, ignoring confidence."""
    y_true = np.concatenate(
        [np.ones(len(pos_verdicts), dtype=np.int32), np.zeros(len(neg_verdicts), dtype=np.int32)]
    )
    y_pred = np.array(
        [1 if v["match"] == "yes" else 0 for v in pos_verdicts + neg_verdicts],
        dtype=np.int32,
    )
    conf = np.array([v["confidence"] for v in pos_verdicts + neg_verdicts], dtype=np.float64)
    correct = y_pred == y_true

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(np.mean(correct)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion": {
            "true_positive": int(tp),
            "false_positive": int(fp),
            "true_negative": int(tn),
            "false_negative": int(fn),
        },
        # A judge that answers "yes" to nearly everything can look fine on
        # recall while carrying no information — yes_rate makes that visible.
        "yes_rate": float(np.mean(y_pred)),
        "yes_rate_on_positives": float(np.mean(y_pred[y_true == 1])),
        "yes_rate_on_negatives": float(np.mean(y_pred[y_true == 0])),
        "mean_confidence": float(np.mean(conf)),
        "mean_confidence_when_correct": float(np.mean(conf[correct])) if correct.any() else None,
        "mean_confidence_when_wrong": (
            float(np.mean(conf[~correct])) if (~correct).any() else None
        ),
    }


def calibration_buckets(
    pos_verdicts: list[dict[str, Any]],
    neg_verdicts: list[dict[str, Any]],
    edges: tuple[float, ...] = (0.0, 60.0, 70.0, 80.0, 90.0, 100.001),
) -> list[dict[str, Any]]:
    """Accuracy per stated-confidence band — is high confidence actually better?"""
    y_true = np.concatenate(
        [np.ones(len(pos_verdicts), dtype=np.int32), np.zeros(len(neg_verdicts), dtype=np.int32)]
    )
    y_pred = np.array(
        [1 if v["match"] == "yes" else 0 for v in pos_verdicts + neg_verdicts],
        dtype=np.int32,
    )
    conf = np.array([v["confidence"] for v in pos_verdicts + neg_verdicts], dtype=np.float64)
    correct = y_pred == y_true

    out: list[dict[str, Any]] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf < hi)
        n = int(mask.sum())
        out.append(
            {
                "confidence_range": f"[{lo:g}, {hi if hi <= 100 else 100:g}]",
                "n": n,
                "accuracy": float(np.mean(correct[mask])) if n else None,
                "mean_stated_confidence": float(np.mean(conf[mask])) if n else None,
            }
        )
    return out


def intent_pair_auc(
    positives: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
    min_n: int = 5,
) -> dict[str, Any]:
    """Per-intent pair AUC only.

    ``baselines/metrics.py::intent_slice_metrics`` also computes MRR/Recall@10
    and therefore requires either ranks or a shared embedding space — an LLM
    judge has neither (see this package's __init__ docstring), so only the
    pair-AUC half is computed, using the same bucketing and the same
    ``_safe_auc`` so numbers line up with the other baselines' ``pair_auc``.

    Note ``_record_intent`` reads ``searchQuery`` to assign a bucket. That is
    an offline analysis label applied after the fact — the query is never part
    of what the judge was shown.
    """
    pos_intents = [_record_intent(r) for r in positives]
    neg_intents = [_record_intent(r) for r in negatives]
    out: dict[str, Any] = {}
    for bucket in sorted(set(pos_intents) | set(neg_intents)):
        pos_idx = [i for i, b in enumerate(pos_intents) if b == bucket]
        neg_idx = [i for i, b in enumerate(neg_intents) if b == bucket]
        n_pair = len(pos_idx) + len(neg_idx)
        slice_out: dict[str, Any] = {"n_pairs": n_pair, "low_n": n_pair < min_n}
        if pos_idx and neg_idx and n_pair >= min_n:
            y_true = np.concatenate(
                [np.ones(len(pos_idx), dtype=np.int32), np.zeros(len(neg_idx), dtype=np.int32)]
            )
            y_score = np.concatenate([pos_scores[pos_idx], neg_scores[neg_idx]])
            slice_out["pair_auc"] = _safe_auc(y_true, y_score)
        else:
            slice_out["pair_auc"] = None
        out[bucket] = slice_out
    return out
