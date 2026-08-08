"""Run both arms end to end and write ``artifacts/bilinear_mf/<run_id>/``.

    python -m bilinear_mf.run --backbone voyage_large --run-id mf_voyage_001
    python -m bilinear_mf.run --backbone tfidf --run-id mf_tfidf_001
    python -m bilinear_mf.run --backbone tfidf --arms lsa      # label-free only

Order of operations matters and is deliberate:

1. Encode once.
2. ``lsa`` sweep — label-free, so nothing here can leak.
3. Inner seeker-disjoint CV **on the 131 train pairs only** to pick the head's
   rank and weight decay. The holdout is not consulted.
4. One-shot holdout eval with those hyperparameters.
5. Seeker-disjoint CV over all 200 for the population that actually
   discriminates between strong models.
6. Label-permutation null on the same CV protocol, so step 5 has a floor.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from bilinear_mf.config import BilinearConfig
from bilinear_mf.evaluate import (
    _headline,
    assemble_metrics,
    cosine_baseline_metrics,
    cv_auc,
    cv_pair_scores,
    cv_score_matrix,
    subset_corpus,
)
from bilinear_mf.features import build_features
from bilinear_mf.model import SvdReducer, cosine_pair_scores, train_bilinear


def _idx_for(features, subset: str | None) -> np.ndarray:
    if subset is None:
        return np.arange(len(features.labels))
    return np.asarray([i for i, s in enumerate(features.subsets) if s == subset])


# ---------------------------------------------------------------------------
# Arm 1 — label-free text matrix factorization
# ---------------------------------------------------------------------------


def run_lsa_arm(features, cfg: BilinearConfig) -> dict[str, Any]:
    """Cosine in the rank-``k`` SVD space, swept over ``k``, plus the un-reduced
    reference so the cost of compression is visible rather than assumed."""
    all_idx = _idx_for(features, None)
    train_idx = _idx_for(features, "train")
    holdout_idx = _idx_for(features, "holdout")
    stacked = np.vstack([features.seeker_emb, features.cand_emb, features.corpus_emb])

    out: dict[str, Any] = {"ranks": {}}

    full = cosine_baseline_metrics(
        features, features.seeker_emb, features.cand_emb, all_idx
    )
    full_holdout = cosine_baseline_metrics(
        features, features.seeker_emb, features.cand_emb, holdout_idx
    )
    full_train = cosine_baseline_metrics(
        features, features.seeker_emb, features.cand_emb, train_idx
    )
    out["no_reduction"] = {
        "dim": features.dim,
        "all": _headline(full),
        "train": _headline(full_train),
        "holdout": _headline(full_holdout),
    }
    print(
        f"  [lsa] no reduction (d={features.dim}): "
        f"all AUC {full['pair']['roc_auc']:.4f} MRR {full['retrieval']['mrr']:.4f}"
    )

    for k in cfg.lsa_ranks:
        if k >= min(stacked.shape):
            continue
        reducer = SvdReducer(n_components=k, seed=cfg.seed).fit(stacked)
        seeker_red = reducer.transform(features.seeker_emb)
        cand_red = reducer.transform(features.cand_emb)
        m_all = cosine_baseline_metrics(features, seeker_red, cand_red, all_idx)
        m_train = cosine_baseline_metrics(features, seeker_red, cand_red, train_idx)
        m_hold = cosine_baseline_metrics(features, seeker_red, cand_red, holdout_idx)
        out["ranks"][str(k)] = {
            "explained_variance_ratio": reducer.explained_variance_ratio_,
            "all": _headline(m_all),
            # `train` exists so `k` can be chosen without consulting either eval
            # population. Sweeping k and reporting the best all-200 number would
            # be selection on the test set — the rank has to be picked here.
            "train": _headline(m_train),
            "holdout": _headline(m_hold),
        }
        print(
            f"  [lsa] k={k:>4} evr={reducer.explained_variance_ratio_:.3f}  "
            f"train AUC {m_train['pair']['roc_auc']:.4f} | "
            f"all AUC {m_all['pair']['roc_auc']:.4f} MRR {m_all['retrieval']['mrr']:.4f}"
        )
    return out


# ---------------------------------------------------------------------------
# Arm 2 — low-rank bilinear head
# ---------------------------------------------------------------------------


def run_bilinear_arm(features, cfg: BilinearConfig, *, n_folds: int) -> dict[str, Any]:
    all_idx = _idx_for(features, None)
    train_idx = _idx_for(features, "train")
    holdout_idx = _idx_for(features, "holdout")

    stacked = np.vstack([features.seeker_emb, features.cand_emb, features.corpus_emb])
    center = features.backbone != "tfidf"

    def reduced(dim: int) -> tuple[np.ndarray, np.ndarray, Any]:
        r = SvdReducer(n_components=dim, seed=cfg.seed).fit(stacked, center=center)
        return r.transform(features.seeker_emb), r.transform(features.cand_emb), r

    base_kwargs = dict(
        lr=cfg.lr, steps=cfg.steps, init_scale=cfg.init_scale, seed=cfg.seed
    )

    # --- 3. inner CV on the train split only -----------------------------
    print(f"\n  [bilinear] inner CV grid on {len(train_idx)} train pairs")
    grid: list[dict[str, Any]] = []
    spaces: dict[int, tuple[np.ndarray, np.ndarray, Any]] = {}
    for dim in cfg.grid_reduce_dims:
        if dim >= min(stacked.shape):
            continue
        spaces[dim] = reduced(dim)
        s_red, c_red, _ = spaces[dim]
        for rank in cfg.grid_ranks:
            for wd in cfg.grid_weight_decay:
                auc = cv_auc(
                    features,
                    s_red,
                    c_red,
                    train_idx,
                    n_folds=n_folds,
                    rank=rank,
                    weight_decay=wd,
                    **base_kwargs,
                )
                grid.append(
                    {
                        "reduce_dim": dim,
                        "rank": rank,
                        "weight_decay": wd,
                        "inner_cv_auc": auc,
                    }
                )
                print(
                    f"    d={dim:>4} rank={rank:>3} wd={wd:<6} inner CV AUC {auc:.4f}"
                )
    best = max(grid, key=lambda g: g["inner_cv_auc"])
    print(
        f"  [bilinear] selected d={best['reduce_dim']} rank={best['rank']} "
        f"wd={best['weight_decay']}"
    )

    seeker_red, cand_red, reducer = spaces[best["reduce_dim"]]
    fit_kwargs = dict(rank=best["rank"], weight_decay=best["weight_decay"], **base_kwargs)

    # --- 4. one-shot holdout ---------------------------------------------
    model = train_bilinear(
        seeker_red[train_idx], cand_red[train_idx], features.labels[train_idx], **fit_kwargs
    )
    hold_sorted = np.asarray(sorted(int(i) for i in holdout_idx))
    hold_metrics = assemble_metrics(
        features,
        hold_sorted,
        model.pair_scores(seeker_red[hold_sorted], cand_red[hold_sorted]),
        score_matrix_fn=model.score_matrix,
        seeker_space=seeker_red,
        cand_space=cand_red,
    )
    hold_cosine = cosine_baseline_metrics(features, seeker_red, cand_red, hold_sorted)
    print(
        f"  [bilinear] holdout: AUC {hold_metrics['pair']['roc_auc']:.4f} "
        f"(cosine {hold_cosine['pair']['roc_auc']:.4f})"
    )

    # --- 5. seeker-disjoint CV over all 200 ------------------------------
    cv_scores, models = cv_pair_scores(
        features, seeker_red, cand_red, all_idx, n_folds=n_folds, **fit_kwargs
    )
    corpus_ids, corpus_red = subset_corpus(features, all_idx, cand_red)
    matrix = cv_score_matrix(
        features, seeker_red, cand_red, all_idx, corpus_ids, corpus_red, models,
        n_folds=n_folds,
    )
    cv_metrics = assemble_metrics(
        features,
        all_idx,
        cv_scores[all_idx],
        score_matrix_fn=lambda q, c: matrix,
        seeker_space=seeker_red,
        cand_space=cand_red,
    )
    cv_cosine = cosine_baseline_metrics(features, seeker_red, cand_red, all_idx)
    print(
        f"  [bilinear] all-200 CV: AUC {cv_metrics['pair']['roc_auc']:.4f} "
        f"(cosine {cv_cosine['pair']['roc_auc']:.4f})"
    )

    # per-fold spread, so a pooled number can't hide a wide one
    from sklearn.metrics import roc_auc_score

    from bilinear_mf.evaluate import seeker_folds

    fold_aucs: list[float] = []
    for fold in seeker_folds(features.seeker_ids, all_idx, n_folds):
        y = features.labels[fold]
        if len(set(y.tolist())) < 2:
            continue
        fold_aucs.append(float(roc_auc_score(y, cv_scores[fold])))

    # --- 6. label-permutation null ---------------------------------------
    print(f"  [bilinear] permutation null, {cfg.n_permutations} draws")
    rng = np.random.default_rng(cfg.seed)
    null: list[float] = []
    t0 = time.time()
    for n in range(cfg.n_permutations):
        shuffled = features.labels.copy()
        perm = rng.permutation(all_idx)
        shuffled[all_idx] = features.labels[perm]
        null.append(
            cv_auc(
                features,
                seeker_red,
                cand_red,
                all_idx,
                labels=shuffled,
                n_folds=n_folds,
                **fit_kwargs,
            )
        )
        if (n + 1) % 10 == 0:
            print(f"    {n + 1}/{cfg.n_permutations} ({time.time() - t0:.0f}s)")
    null_arr = np.asarray(null)
    observed = cv_metrics["pair"]["roc_auc"]
    p_value = float((null_arr >= observed).sum() + 1) / (len(null_arr) + 1)

    return {
        "reduce_dim": best["reduce_dim"],
        "explained_variance_ratio": reducer.explained_variance_ratio_,
        "grid": grid,
        "selected": {
            "reduce_dim": best["reduce_dim"],
            "rank": best["rank"],
            "weight_decay": best["weight_decay"],
            "inner_cv_auc": best["inner_cv_auc"],
        },
        "model": model.to_json(),
        "holdout": {
            "bilinear": _headline(hold_metrics),
            "cosine_same_space": _headline(hold_cosine),
            "full": hold_metrics,
        },
        "cv_all200": {
            "n_folds": n_folds,
            "bilinear": _headline(cv_metrics),
            "cosine_same_space": _headline(cv_cosine),
            "fold_auc_mean": float(np.mean(fold_aucs)) if fold_aucs else None,
            "fold_auc_std": float(np.std(fold_aucs)) if fold_aucs else None,
            "n_folds_scored": len(fold_aucs),
            "full": cv_metrics,
        },
        "permutation_null": {
            "n": len(null_arr),
            "mean": float(null_arr.mean()),
            "std": float(null_arr.std()),
            "p95": float(np.percentile(null_arr, 95)),
            "max": float(null_arr.max()),
            "observed": observed,
            "p_value": p_value,
        },
    }


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default="mf_001")
    ap.add_argument("--backbone", choices=["tfidf", "voyage_large"], default="voyage_large")
    ap.add_argument("--arms", default="lsa,bilinear", help="comma-separated subset")
    ap.add_argument("--data-dir", type=Path, default=BilinearConfig.data_dir)
    ap.add_argument("--split-path", type=Path, default=BilinearConfig.split_path)
    ap.add_argument("--reduce-dim", type=int, default=BilinearConfig.reduce_dim)
    ap.add_argument("--steps", type=int, default=BilinearConfig.steps)
    ap.add_argument("--n-folds", type=int, default=10)
    ap.add_argument("--n-permutations", type=int, default=BilinearConfig.n_permutations)
    args = ap.parse_args()

    cfg = BilinearConfig(
        run_id=args.run_id,
        backbone=args.backbone,
        data_dir=args.data_dir,
        split_path=args.split_path,
        reduce_dim=args.reduce_dim,
        steps=args.steps,
        n_permutations=args.n_permutations,
    )
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    print(f"=== bilinear_mf {cfg.run_id} | backbone={cfg.backbone} | arms={arms} ===")
    features = build_features(
        data_dir=cfg.data_dir,
        split_path=cfg.split_path,
        backbone=cfg.backbone,
        max_features=cfg.max_features,
        ngram_range=cfg.ngram_range,
        voyage_model=cfg.voyage_model,
        voyage_output_dimension=cfg.voyage_output_dimension,
    )
    n_train = sum(1 for s in features.subsets if s == "train")
    print(
        f"encoded {len(features.labels)} pairs (d={features.dim}), "
        f"corpus {len(features.corpus_ids)}, train {n_train} / "
        f"holdout {len(features.labels) - n_train}, "
        f"seekers {len(set(features.seeker_ids))}"
    )

    results: dict[str, Any] = {
        "run_id": cfg.run_id,
        "config": cfg.to_json(),
        "real_data_hash": features.real_data_hash,
        "n_pairs": int(len(features.labels)),
        "n_candidates": len(features.corpus_ids),
        "n_seekers": len(set(features.seeker_ids)),
        "backbone_dim": features.dim,
    }
    if "lsa" in arms:
        print("\n--- arm: lsa (label-free text matrix factorization) ---")
        results["lsa"] = run_lsa_arm(features, cfg)
    if "bilinear" in arms:
        print("\n--- arm: bilinear (low-rank scoring-function factorization) ---")
        results["bilinear"] = run_bilinear_arm(features, cfg, n_folds=args.n_folds)

    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.run_dir / "results.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
