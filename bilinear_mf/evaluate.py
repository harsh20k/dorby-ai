"""Scoring protocols, and the discipline that keeps their numbers honest.

Three separate questions, three separate protocols:

``lsa_sweep``
    Label-free, so it can be scored on all 200 pairs directly and compared to
    any row in ``docs/baseline-results-real200.md``.

``holdout``
    The bilinear head trained on the 131 frozen train pairs, scored once on the
    69-pair holdout. Clean, but a 69-pair readout — the population this repo has
    already documented as reversing conclusions three times.

``cv``
    Seeker-disjoint leave-one-group-out over all 200 pairs, each pair scored by
    a model that never saw its seeker. This is the only way to get an honest
    all-200 number for a *trained* model — ``eval_real_full/guard.py`` exists
    precisely to stop a train-contaminated adapter being scored there, and a
    naive "train on train, score on all" would be exactly that flattering
    mistake.

Every CV result is reported against a **label-permutation null**: the identical
protocol re-run on shuffled labels, many times. With 131 training pairs and a
head of a few thousand parameters, the question is never "is the AUC above 0.5"
but "is it above what this pipeline scores on noise". ``moe_reranker`` learned
that the expensive way (fold std 0.067 swamping every effect it measured).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from baselines.metrics import (
    pair_metrics,
    ranks_from_score_matrix,
    retrieval_metrics_from_ranks,
    slice_metrics,
)

from bilinear_mf.features import PairFeatures
from bilinear_mf.model import BilinearScorer, cosine_pair_scores, train_bilinear


def subset_corpus(
    features: PairFeatures, idx: np.ndarray, cand_space: np.ndarray | None = None
) -> tuple[list[str], np.ndarray]:
    """First-seen unique candidates over ``idx``, reusing already-encoded rows.

    Reproduces ``features.build_candidate_corpus``'s ordering over a subset
    without re-encoding: pair rows are stored positives-then-negatives, which is
    the order that function walks, so ascending ``idx`` yields the same corpus.

    ``cand_space`` selects which representation the corpus is drawn from — the
    raw backbone, or a reduced space. It must be passed whenever ranking happens
    somewhere other than the backbone, or the retrieval metrics silently
    describe a different model than the pair metrics do.
    """
    space = features.cand_emb if cand_space is None else cand_space
    all_ids = features.pos_target_ids + [
        r["matchContactId"] for r in features.negatives
    ]
    ids: list[str] = []
    rows: list[int] = []
    seen: set[str] = set()
    for i in sorted(int(j) for j in idx):
        cid = all_ids[i]
        if cid in seen:
            continue
        seen.add(cid)
        ids.append(cid)
        rows.append(i)
    return ids, space[rows]


def assemble_metrics(
    features: PairFeatures,
    idx: np.ndarray,
    pair_scores: np.ndarray,
    *,
    score_matrix_fn,
    seeker_space: np.ndarray | None = None,
    cand_space: np.ndarray | None = None,
) -> dict[str, Any]:
    """The same four metric calls, in the same order, as every baseline eval.

    ``pair_scores`` must be aligned to ``idx`` sorted ascending.

    ``score_matrix_fn(queries, corpus)`` produces the (n_queries x n_candidates)
    matrix. Passed as a callable because the bilinear arm does not rank by
    dot-product, so ``retrieval_metrics`` (which assumes cosine) cannot be used —
    the score-matrix path is the one the late-fusion hybrids already take.

    ``seeker_space``/``cand_space`` default to the raw backbone. Both must be
    supplied together with a matching ``score_matrix_fn`` when the model scores
    in a reduced space; passing only one silently ranks in the wrong space.
    """
    idx = np.asarray(sorted(int(i) for i in idx))
    labels = features.labels[idx]
    pos_local = idx[labels == 1]
    neg_local = idx[labels == 0]

    pos_scores = pair_scores[labels == 1]
    neg_scores = pair_scores[labels == 0]

    n_pos_all = len(features.positives)
    positives = [features.positives[i] for i in pos_local]
    negatives = [features.negatives[i - n_pos_all] for i in neg_local]
    neg_seeker_texts = [features.neg_seeker_texts[i - n_pos_all] for i in neg_local]
    neg_cand_texts = [features.neg_cand_texts[i - n_pos_all] for i in neg_local]
    pos_target_ids = [features.pos_target_ids[i] for i in pos_local]

    seekers = features.seeker_emb if seeker_space is None else seeker_space
    corpus_ids, corpus_emb = subset_corpus(features, idx, cand_space)
    matrix = score_matrix_fn(seekers[pos_local], corpus_emb)
    ranks = ranks_from_score_matrix(matrix, pos_target_ids, corpus_ids)

    return {
        "n_pairs": int(len(idx)),
        "n_candidates": len(corpus_ids),
        "pair": pair_metrics(pos_scores, neg_scores),
        "retrieval": retrieval_metrics_from_ranks(ranks),
        "slices": slice_metrics(
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
        ),
    }


def _headline(metrics: dict[str, Any]) -> dict[str, float]:
    hard = metrics["slices"]["neg_hardness"]

    def _auc(bucket: str) -> float | None:
        entry = hard.get(bucket) or {}
        return entry.get("pair_auc")

    return {
        "pair_auc": metrics["pair"]["roc_auc"],
        "hard_neg_auc": _auc("hard"),
        "easy_neg_auc": _auc("easy"),
        "mrr": metrics["retrieval"]["mrr"],
        "recall@1": metrics["retrieval"]["recall@1"],
        "recall@10": metrics["retrieval"]["recall@10"],
    }


# ---------------------------------------------------------------------------
# Cross-validation over seekers
# ---------------------------------------------------------------------------


def seeker_folds(
    seeker_ids: list[str], idx: np.ndarray, n_folds: int = 10
) -> list[np.ndarray]:
    """Seeker-disjoint K-fold: every pair of a given seeker lands in one fold.

    Grouping by seeker, not by pair, is load-bearing. `rrf_002`'s probes found
    seeker identity alone predicts a label at 0.687 AUC; a random pair split
    would let a model memorize "this seeker declines everything" and report it
    as matching skill.

    K-fold rather than leave-one-seeker-out purely for cost: the permutation
    null re-runs this whole protocol 50 times, and 129 single-seeker folds would
    make that ~6,500 trainings for no extra rigor. Assignment is deterministic
    (seekers sorted, dealt round-robin) so a run is reproducible.
    """
    by_seeker: dict[str, list[int]] = {}
    for i in idx:
        by_seeker.setdefault(seeker_ids[int(i)], []).append(int(i))
    buckets: list[list[int]] = [[] for _ in range(max(1, n_folds))]
    for n, (_, rows) in enumerate(sorted(by_seeker.items())):
        buckets[n % len(buckets)].extend(rows)
    return [np.asarray(sorted(b)) for b in buckets if b]


def cv_pair_scores(
    features: PairFeatures,
    seeker_red: np.ndarray,
    cand_red: np.ndarray,
    idx: np.ndarray,
    *,
    labels: np.ndarray | None = None,
    n_folds: int = 10,
    rank: int,
    weight_decay: float,
    lr: float,
    steps: int,
    init_scale: float,
    seed: int,
) -> tuple[np.ndarray, list[BilinearScorer]]:
    """Out-of-fold scores for every pair in ``idx``, plus the per-fold models.

    ``labels`` overrides the real labels — that hook is how the permutation null
    reuses this exact code path rather than an approximation of it.
    """
    y = features.labels if labels is None else labels
    folds = seeker_folds(features.seeker_ids, idx, n_folds)
    out = np.zeros(len(features.labels), dtype=np.float32)
    models: list[BilinearScorer] = []

    for fold in folds:
        held = set(int(i) for i in fold)
        train_idx = np.asarray([int(i) for i in idx if int(i) not in held])
        model = train_bilinear(
            seeker_red[train_idx],
            cand_red[train_idx],
            y[train_idx],
            rank=rank,
            weight_decay=weight_decay,
            lr=lr,
            steps=steps,
            init_scale=init_scale,
            seed=seed,
        )
        out[fold] = model.pair_scores(seeker_red[fold], cand_red[fold])
        models.append(model)
    return out, models


def cv_score_matrix(
    features: PairFeatures,
    seeker_red: np.ndarray,
    cand_red: np.ndarray,
    idx: np.ndarray,
    corpus_ids: list[str],
    corpus_emb: np.ndarray,
    models: list[BilinearScorer],
    n_folds: int = 10,
) -> np.ndarray:
    """Retrieval scores where each query's row comes from its own fold's model.

    Ranking a query with a model that trained on that query's seeker would
    inflate MRR exactly the way scoring pairs in-fold inflates AUC, so the fold
    structure has to be carried through to the retrieval matrix too — not just
    the pair scores.
    """
    folds = seeker_folds(features.seeker_ids, idx, n_folds)
    pos_idx = np.asarray(sorted(int(i) for i in idx if features.labels[int(i)] == 1))
    row_of = {int(g): r for r, g in enumerate(pos_idx)}
    matrix = np.zeros((len(pos_idx), len(corpus_ids)), dtype=np.float32)

    for fold, model in zip(folds, models):
        rows = [int(i) for i in fold if features.labels[int(i)] == 1]
        if not rows:
            continue
        block = model.score_matrix(seeker_red[rows], corpus_emb)
        for r, global_i in enumerate(rows):
            matrix[row_of[global_i]] = block[r]
    return matrix


def cv_auc(
    features: PairFeatures,
    seeker_red: np.ndarray,
    cand_red: np.ndarray,
    idx: np.ndarray,
    *,
    labels: np.ndarray | None = None,
    **train_kwargs,
) -> float:
    """Out-of-fold pooled ROC-AUC — the selection criterion for the inner grid."""
    from sklearn.metrics import roc_auc_score

    y = features.labels if labels is None else labels
    scores, _ = cv_pair_scores(
        features, seeker_red, cand_red, idx, labels=labels, **train_kwargs
    )
    return float(roc_auc_score(y[idx], scores[idx]))


def cosine_baseline_metrics(
    features: PairFeatures,
    seeker_red: np.ndarray,
    cand_red: np.ndarray,
    idx: np.ndarray,
) -> dict[str, Any]:
    """Frozen cosine in the same reduced space — the number the head must beat."""
    idx_sorted = np.asarray(sorted(int(i) for i in idx))
    scores = cosine_pair_scores(seeker_red[idx_sorted], cand_red[idx_sorted])
    return assemble_metrics(
        features,
        idx_sorted,
        scores,
        score_matrix_fn=lambda q, c: q @ c.T,
        seeker_space=seeker_red,
        cand_space=cand_red,
    )
