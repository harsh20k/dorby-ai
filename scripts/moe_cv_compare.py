#!/usr/bin/env python3
"""Does the MMoE earn its parameters? Seeker-disjoint cross-validation.

Why this exists: a single 20-pair train-dev split (14 pos / 6 neg) has an AUC
granularity of 1/84 and cannot adjudicate between models. So instead of one
split, this runs seeker-disjoint K-fold CV over the 111-pair train pool and
compares four models on **identical features**:

  1. ``logistic``      — plain logistic regression. The control. If this wins,
                         the MoE machinery is not paying for itself.
  2. ``moe_single``    — MoE with the auxiliary task off (``aux_weight=0``).
                         Isolates the "mixture of experts" part.
  3. ``moe_multi``     — full MMoE, judge-score auxiliary task on. Isolates the
                         "multi-gate / multi-task" part, which is the actual
                         claim of the architecture.
  4. ``nano_cosine``   — the single raw feature, no model at all. The floor.

Reporting mean AUC across folds with its spread is the point: differences
smaller than the fold-to-fold standard deviation are not results.

    PYTHONPATH=. .venv/bin/python scripts/moe_cv_compare.py --folds 5
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from baselines.metrics import pair_metrics
from moe_reranker.config import MoEConfig
from moe_reranker.data import load
from moe_reranker.model import MMoE, balance_loss, sharpen_loss, task_loss


def seeker_disjoint_folds(seeker_ids: list[str], k: int, seed: int) -> list[np.ndarray]:
    """Assign whole seekers to folds, so no seeker spans train and validation."""
    rng = np.random.default_rng(seed)
    uniq = sorted(set(seeker_ids))
    rng.shuffle(uniq)
    fold_of = {s: i % k for i, s in enumerate(uniq)}
    assign = np.array([fold_of[s] for s in seeker_ids])
    return [np.where(assign == f)[0] for f in range(k)]


def _auc(scores: np.ndarray, y: np.ndarray) -> float:
    pos, neg = scores[y > 0.5], scores[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(pair_metrics(pos, neg)["roc_auc"])


def fit_moe(
    cfg: MoEConfig,
    Xtr: np.ndarray,
    Ttr: np.ndarray,
    Mtr: np.ndarray,
    Xva: np.ndarray,
    *,
    epochs: int,
) -> np.ndarray:
    torch.manual_seed(cfg.seed)
    model = MMoE(
        n_features=Xtr.shape[1],
        n_experts=cfg.n_experts,
        expert_hidden=cfg.expert_hidden,
        n_tasks=len(cfg.task_names),
        tau=cfg.tau,
        expert_dropout=cfg.expert_dropout,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    xt, tt, mt = torch.from_numpy(Xtr), torch.from_numpy(Ttr), torch.from_numpy(Mtr)

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(perm), cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            logits, gates = model(xt[idx])
            loss = (
                task_loss(logits, tt[idx], mt[idx], cfg.task_weights)
                + cfg.sharpen_weight * sharpen_loss(gates)
                + cfg.balance_weight * balance_loss(gates)
            )
            loss.backward()
            opt.step()
    return model.score(torch.from_numpy(Xva)).numpy()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Fixed budget per fold. Kept small on purpose: the single-split run "
        "showed dev AUC peaking at epoch 1 and falling steadily after.",
    )
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--aux-weight", type=float, default=0.3)
    p.add_argument(
        "--out", type=Path, default=Path("artifacts/moe_reranker/cv_compare.json")
    )
    a = p.parse_args()

    base = replace(MoEConfig(), tau=a.tau, seed=a.seed)
    splits = load(base)
    tr = splits["train"]
    X, y = tr.X, tr.y
    T, M = tr.targets()
    nano_idx = 0  # 'nano_cos' is feature 0; see features.FEATURE_NAMES

    folds = seeker_disjoint_folds(tr.seeker_ids, a.folds, a.seed)
    print(
        f"{len(y)} train pairs, {len(set(tr.seeker_ids))} seekers, "
        f"{a.folds} seeker-disjoint folds, {a.epochs} epochs/fold"
    )

    from sklearn.linear_model import LogisticRegression

    results: dict[str, list[float]] = {
        "nano_cosine": [],
        "logistic": [],
        "moe_single": [],
        "moe_multi": [],
    }

    for f, va_idx in enumerate(folds):
        tr_idx = np.setdiff1d(np.arange(len(y)), va_idx)
        if len(np.unique(y[va_idx])) < 2 or len(np.unique(y[tr_idx])) < 2:
            print(f"  fold {f}: skipped (single-class split)")
            continue
        Xtr, Xva = X[tr_idx], X[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]

        results["nano_cosine"].append(_auc(Xva[:, nano_idx], yva))

        lr = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr)
        results["logistic"].append(_auc(lr.predict_proba(Xva)[:, 1], yva))

        single = replace(base, task_weights=(1.0, 0.0))
        results["moe_single"].append(
            _auc(fit_moe(single, Xtr, T[tr_idx], M[tr_idx], Xva, epochs=a.epochs), yva)
        )

        multi = replace(base, task_weights=(1.0, a.aux_weight))
        results["moe_multi"].append(
            _auc(fit_moe(multi, Xtr, T[tr_idx], M[tr_idx], Xva, epochs=a.epochs), yva)
        )

        print(
            f"  fold {f} (n_val={len(va_idx)}): "
            + "  ".join(f"{k} {v[-1]:.3f}" for k, v in results.items())
        )

    print(f"\n{'model':<14} {'mean AUC':>9} {'std':>7} {'min':>7} {'max':>7}")
    print("-" * 48)
    summary: dict[str, Any] = {}
    for name, vals in results.items():
        arr = np.array(vals, dtype=float)
        summary[name] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "min": float(arr.min()),
            "max": float(arr.max()),
            "folds": [float(v) for v in arr],
        }
        print(
            f"{name:<14} {arr.mean():>9.4f} {summary[name]['std']:>7.4f} "
            f"{arr.min():>7.4f} {arr.max():>7.4f}"
        )

    best = max(summary, key=lambda k: summary[k]["mean"])
    print(f"\nbest mean AUC: {best} ({summary[best]['mean']:.4f})")
    spread = summary[best]["std"]
    close = [
        k
        for k in summary
        if k != best and summary[best]["mean"] - summary[k]["mean"] < spread
    ]
    if close:
        print(
            "  within one fold-to-fold std of the winner (i.e. not separable): "
            + ", ".join(close)
        )

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(
        json.dumps(
            {
                "folds": a.folds,
                "epochs": a.epochs,
                "tau": a.tau,
                "aux_weight": a.aux_weight,
                "n_train": len(y),
                "n_seekers": len(set(tr.seeker_ids)),
                "summary": summary,
            },
            indent=2,
        )
    )
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
