"""Does training on 2,619 synthetic judge-labeled pairs help on real pairs?

    PYTHONPATH=. .venv/bin/python -m moe_rrf.experiment
    PYTHONPATH=. .venv/bin/python -m moe_rrf.experiment --holdout   # one-shot, only if an arm wins

The question the previous experiment left open: the MMoE was untestable on 111
real pairs, so the bottleneck looked like data rather than architecture. rrf_003
supplies 2,619 pairs and 2,773 within-seeker triplets. Does that change anything
measured on *real* pairs?

Arms (all on the same 12 text-only features, see ``features.py``):

  ``tfidf_only``     TF-IDF cosine alone, no model. The floor.
  ``logistic_real``  logistic regression, real pairs, seeker-disjoint CV.
  ``moe_real``       MMoE, real pairs only, seeker-disjoint CV. Reproduces the
                     earlier null on this feature set.
  ``moe_synth``      MMoE trained on synthetic pairs only, evaluated on all 131
                     real eval-pool pairs. Zero real pairs seen in training, so
                     the whole pool is a clean test set.
  ``moe_synth_ws``   Same, but trained on **within-seeker triplets** — a ranking
                     loss on (anchor, pos, neg) that cancels the per-seeker base
                     rate by construction. This is the arm that tests what the
                     synthetic batch actually unlocks.
  ``logistic_synth`` logistic regression trained on synthetic, evaluated on real.
                     The control that says whether any gain is about the MoE or
                     just about having more data.
  ``moe_transfer``   Pretrain on synthetic, fine-tune on real train folds,
                     evaluate on the held-out real fold (CV).

Every arm is scored on real pairs. Synthetic-internal accuracy is reported too,
but only as a sanity check that the model learned *something* — it cannot be the
headline, because it measures agreement with a judge that is 0.5942-accurate on
hard real pairs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from baselines.metrics import pair_metrics
from moe_reranker import diagnostics
from moe_reranker.model import MMoE, balance_loss, sharpen_loss
from moe_rrf import data as D
from moe_rrf.features import FEATURE_NAMES, Standardizer, build_raw, tfidf_channel

TAU = 0.05
N_EXPERTS = 3
EXPERT_HIDDEN = 4
EXPERT_DROPOUT = 0.2
SHARPEN_W = 0.05
BALANCE_W = 0.10
LR = 3e-3
WEIGHT_DECAY = 1e-2
BATCH = 64


def auc(scores: np.ndarray, y: np.ndarray) -> float:
    pos, neg = scores[y > 0.5], scores[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(pair_metrics(pos, neg)["roc_auc"])


def featurize(
    train_rows: list[dict[str, Any]], eval_sets: dict[str, list[dict[str, Any]]]
) -> tuple[np.ndarray, dict[str, np.ndarray], Standardizer]:
    """Fit TF-IDF + standardizer on the TRAINING population, apply to each eval set."""
    tr_cos, tr_rank = tfidf_channel(train_rows, train_rows)
    tr_raw = build_raw(train_rows, tfidf_cos=tr_cos, tfidf_rank_pct=tr_rank)
    std = Standardizer().fit(tr_raw)

    out: dict[str, np.ndarray] = {}
    for name, rows in eval_sets.items():
        cos, rank = tfidf_channel(rows, train_rows)  # vocabulary from train only
        out[name] = std.transform(build_raw(rows, tfidf_cos=cos, tfidf_rank_pct=rank))
    return std.transform(tr_raw), out, std


def _new_model(n_features: int, seed: int) -> MMoE:
    torch.manual_seed(seed)
    return MMoE(
        n_features=n_features,
        n_experts=N_EXPERTS,
        expert_hidden=EXPERT_HIDDEN,
        n_tasks=1,  # single task here; the auxiliary judge head is the label itself
        tau=TAU,
        expert_dropout=EXPERT_DROPOUT,
    )


def _gate_terms(gates: torch.Tensor) -> torch.Tensor:
    return SHARPEN_W * sharpen_loss(gates) + BALANCE_W * balance_loss(gates)


def train_pairwise(
    X: np.ndarray, y: np.ndarray, *, epochs: int, seed: int, model: MMoE | None = None
) -> MMoE:
    """Plain per-pair BCE."""
    m = model or _new_model(X.shape[1], seed)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    xt = torch.from_numpy(X)
    yt = torch.from_numpy(y).unsqueeze(1)
    lossf = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(perm), BATCH):
            idx = perm[i : i + BATCH]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            logits, gates = m(xt[idx])
            (lossf(logits, yt[idx]) + _gate_terms(gates)).backward()
            opt.step()
    return m


def train_within_seeker(
    X: np.ndarray,
    triplets: list[tuple[int, int]],
    *,
    epochs: int,
    seed: int,
    max_triplets: int = 20000,
) -> MMoE:
    """Rank the positive above the negative *within the same seeker*.

    Loss is BCE on the score difference, which depends only on the ordering inside
    a seeker's own candidates. A seeker who rejects everything contributes no
    triplets at all, so the per-seeker base rate cannot be learned from this
    objective — that is the entire point.
    """
    rng = np.random.default_rng(seed)
    trips = np.array(triplets)
    if len(trips) > max_triplets:
        trips = trips[rng.choice(len(trips), max_triplets, replace=False)]

    m = _new_model(X.shape[1], seed)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    xt = torch.from_numpy(X)
    lossf = torch.nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        m.train()
        order = rng.permutation(len(trips))
        for i in range(0, len(order), BATCH):
            batch = trips[order[i : i + BATCH]]
            if len(batch) < 2:
                continue
            opt.zero_grad()
            lp, gp = m(xt[batch[:, 0]])
            ln, gn = m(xt[batch[:, 1]])
            diff = lp - ln
            loss = lossf(diff, torch.ones_like(diff)) + _gate_terms(
                torch.cat([gp, gn], dim=0)
            )
            loss.backward()
            opt.step()
    return m


def score(m: MMoE, X: np.ndarray) -> np.ndarray:
    m.eval()
    with torch.no_grad():
        logits, _ = m(torch.from_numpy(X))
    return torch.sigmoid(logits[:, 0]).numpy()


def run(args: argparse.Namespace) -> dict[str, Any]:
    real_pool, real_holdout = D.load_real()
    synth = D.load_synth(args.batch_id)
    D.assert_disjoint(synth, real_pool, real_holdout)

    print(real_pool.summary())
    print(real_holdout.summary())
    print(synth.summary())

    synth_trips = D.within_seeker_triplets(synth)
    real_trips = D.within_seeker_triplets(real_pool)
    print(
        f"\nwithin-seeker triplets: synth {len(synth_trips)}, real {len(real_trips)}"
        f"  ({len(synth_trips) / max(1, len(real_trips)):.0f}x)"
    )
    print("disjointness asserted: no contact id shared between populations\n")

    results: dict[str, Any] = {}

    # ---------------- arms trained on SYNTH, evaluated on the full real pool ----
    Xs, ev, _ = featurize(
        synth.rows, {"real": real_pool.rows, "holdout": real_holdout.rows}
    )
    Xr_from_synth, Xh_from_synth = ev["real"], ev["holdout"]

    # Two TF-IDF floors, and the gap between them is a result in its own right.
    # `synthfit` uses a vocabulary/IDF learned from synthetic profiles; `realfit`
    # learns it from real ones. Same formula, same pairs scored — the only
    # difference is which population taught it what words matter. Any gap is
    # distribution shift measured in the feature layer, before any model exists.
    cos_synthfit, _ = tfidf_channel(real_pool.rows, synth.rows)
    cos_realfit, _ = tfidf_channel(real_pool.rows, real_pool.rows)
    results["tfidf_synthfit"] = {
        "real_auc": auc(cos_synthfit, real_pool.y),
        "trained_on": "vocabulary from rrf_003 text",
        "n_train": 0,
    }
    results["tfidf_realfit"] = {
        "real_auc": auc(cos_realfit, real_pool.y),
        "trained_on": "vocabulary from real text",
        "n_train": 0,
    }

    from sklearn.linear_model import LogisticRegression

    lg = LogisticRegression(max_iter=3000).fit(Xs, synth.y)
    results["logistic_synth"] = {
        "real_auc": auc(lg.predict_proba(Xr_from_synth)[:, 1], real_pool.y),
        "synth_train_auc": auc(lg.predict_proba(Xs)[:, 1], synth.y),
        "trained_on": f"{args.batch_id} (pairwise)",
        "n_train": len(synth),
    }

    m_pair = train_pairwise(Xs, synth.y, epochs=args.synth_epochs, seed=args.seed)
    s_real = score(m_pair, Xr_from_synth)
    results["moe_synth"] = {
        "real_auc": auc(s_real, real_pool.y),
        "synth_train_auc": auc(score(m_pair, Xs), synth.y),
        "trained_on": f"{args.batch_id} (pairwise)",
        "n_train": len(synth),
    }

    m_ws = train_within_seeker(
        Xs, synth_trips, epochs=args.synth_epochs, seed=args.seed
    )
    s_real_ws = score(m_ws, Xr_from_synth)
    results["moe_synth_ws"] = {
        "real_auc": auc(s_real_ws, real_pool.y),
        "synth_train_auc": auc(score(m_ws, Xs), synth.y),
        "trained_on": f"{args.batch_id} (within-seeker triplets)",
        "n_train": len(synth_trips),
    }

    # Diagnostics on the within-seeker model, read on the real pool it is judged on.
    diag = diagnostics.compute(
        _gates(m_ws, Xr_from_synth), real_pool.seeker_ids, n_permutations=300
    )
    results["moe_synth_ws"]["diagnostics_on_real"] = diag.as_json()

    # ---------------- arms trained on REAL, seeker-disjoint CV ------------------
    folds = D.seeker_disjoint_folds(real_pool.seeker_ids, args.folds, args.seed)
    cv: dict[str, list[float]] = {"logistic_real": [], "moe_real": [], "moe_transfer": []}
    for f, va in enumerate(folds):
        tr = np.setdiff1d(np.arange(len(real_pool)), va)
        if len(np.unique(real_pool.y[va])) < 2 or len(np.unique(real_pool.y[tr])) < 2:
            continue
        tr_rows = [real_pool.rows[i] for i in tr]
        va_rows = [real_pool.rows[i] for i in va]
        Xtr, evf, _ = featurize(tr_rows, {"va": va_rows})
        Xva = evf["va"]
        ytr, yva = real_pool.y[tr], real_pool.y[va]

        cv["logistic_real"].append(
            auc(
                LogisticRegression(max_iter=3000).fit(Xtr, ytr).predict_proba(Xva)[:, 1],
                yva,
            )
        )
        cv["moe_real"].append(
            auc(
                score(train_pairwise(Xtr, ytr, epochs=args.real_epochs, seed=args.seed), Xva),
                yva,
            )
        )
        # Transfer: start from the synth-pretrained weights, fine-tune on this fold.
        warm = train_within_seeker(Xs, synth_trips, epochs=args.synth_epochs, seed=args.seed)
        cv["moe_transfer"].append(
            auc(
                score(
                    train_pairwise(
                        Xtr, ytr, epochs=args.real_epochs, seed=args.seed, model=warm
                    ),
                    Xva,
                ),
                yva,
            )
        )
        print(f"  fold {f} (n_val={len(va)}): " + "  ".join(
            f"{k} {v[-1]:.3f}" for k, v in cv.items() if v
        ))

    for name, vals in cv.items():
        a = np.array(vals, dtype=float)
        results[name] = {
            "real_auc": float(a.mean()),
            "fold_std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "folds": [float(v) for v in a],
            "trained_on": "real (seeker-disjoint CV)",
            "n_train": len(real_pool) - len(folds[0]),
        }

    # ---------------- report ---------------------------------------------------
    order = [
        "tfidf_realfit",
        "tfidf_synthfit",
        "logistic_real",
        "moe_real",
        "logistic_synth",
        "moe_synth",
        "moe_synth_ws",
        "moe_transfer",
    ]
    print(f"\n{'arm':<16} {'real AUC':>9} {'±std':>7} {'n_train':>8}  trained on")
    print("-" * 78)
    for k in order:
        r = results.get(k)
        if not r:
            continue
        std = f"{r['fold_std']:.3f}" if "fold_std" in r else "   -"
        print(
            f"{k:<16} {r['real_auc']:>9.4f} {std:>7} {r['n_train']:>8}  {r['trained_on']}"
        )

    base = results["moe_real"]["real_auc"]
    best_synth = max(
        ("moe_synth", "moe_synth_ws", "logistic_synth", "moe_transfer"),
        key=lambda k: results[k]["real_auc"],
    )
    delta = results[best_synth]["real_auc"] - base
    noise = results["moe_real"].get("fold_std", 0.0)
    print(
        f"\nbest synth-informed arm: {best_synth} {results[best_synth]['real_auc']:.4f}"
        f"  vs real-only {base:.4f}  ({delta:+.4f}; real-only fold std {noise:.3f})"
    )
    print(
        "  -> "
        + (
            "EXCEEDS the real-only fold-to-fold spread"
            if delta > noise
            else "within the real-only fold-to-fold spread — not separable"
        )
    )

    print("\n" + "=" * 78)
    print(diagnostics.render(diag, ("accept",)))
    print("=" * 78)

    if args.holdout:
        m_best = m_ws if best_synth == "moe_synth_ws" else m_pair
        pm = pair_metrics(
            *(
                lambda s: (s[real_holdout.y > 0.5], s[real_holdout.y <= 0.5])
            )(score(m_best, Xh_from_synth))
        )
        results["holdout"] = {"arm": best_synth, "pair": pm}
        print(f"\nHOLDOUT (one-shot) {best_synth}: pair AUC {pm['roc_auc']:.4f}")
    else:
        print("\nholdout NOT evaluated (--holdout to spend the one-shot check)")

    out_dir = Path("artifacts/moe_rrf") / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": args.run_id,
        "batch_id": args.batch_id,
        "features": FEATURE_NAMES,
        "n_real_eval_pool": len(real_pool),
        "n_real_holdout": len(real_holdout),
        "n_synth": len(synth),
        "synth_triplets": len(synth_trips),
        "real_triplets": len(real_trips),
        "results": results,
    }
    (out_dir / "result.json").write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nwrote {out_dir}/result.json")
    return payload


def _gates(m: MMoE, X: np.ndarray) -> np.ndarray:
    m.eval()
    with torch.no_grad():
        _, g = m(torch.from_numpy(X))
    return g.numpy()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", default="rrf003_001")
    p.add_argument("--batch-id", default="rrf_003")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--synth-epochs", type=int, default=12)
    p.add_argument("--real-epochs", type=int, default=15)
    p.add_argument("--holdout", action="store_true")
    run(p.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
