"""Run the sectioned-MoE arms and report against the standing bar.

Arms, chosen so that each one isolates a single claim:

``logistic_pair``   Logistic regression on the 12 pair scalars. Not a baseline
                    of convenience — it is the thing to beat (0.6398), and it is
                    refit here so the comparison is on identical folds.
``moe_attention``   The recommendation: per-section experts, learned attention
                    pooling.
``moe_softmax``     Same model, pooling frozen at the rule the aggregation sweep
                    already picked. Isolates whether *learning* the pooling helps.
``mlp_attention``   Attention pooling with a single expert. Isolates whether the
                    *mixture* is doing anything the pooling isn't already doing —
                    the redundancy risk called out in the plan.
``moe_mean``        Mean pooling. The null: does treating asks separately help at
                    all, or only treating them *selectively*?

Everything is scored by seeker-disjoint 5-fold CV on the 131-pair train pool.
The frozen 69-pair holdout is not touched unless ``--run-holdout`` is passed
**and** the CV result clears the bar; the runner refuses otherwise.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from moe_rrf.features import build_raw, tfidf_channel
from moe_reranker import diagnostics as diag

from . import data as D
from .config import ENCODER_QWEN3, ENCODER_TFIDF, SectionedConfig
from .encode import Qwen3Backend, TfidfBackend
from .features import RowStandardizer, build_section_features
from .sections import section_stats
from .train import fit, predict


def _make_encoder(cfg: SectionedConfig, fit_texts: list[str]):
    if cfg.encoder == ENCODER_TFIDF:
        return TfidfBackend().fit(fit_texts)
    if cfg.encoder == ENCODER_QWEN3:
        return Qwen3Backend(embedding_dir=cfg.embedding_dir)
    raise ValueError(f"unknown encoder {cfg.encoder!r}")


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")


def run(cfg: SectionedConfig, *, verbose: bool = True) -> dict[str, Any]:
    pool, holdout = D.load_real(cfg.data_dir, cfg.split_path)
    if verbose:
        print(pool.summary())
        print(holdout.summary())
        print("sections:", section_stats(
            pool.rows, min_chars=cfg.min_section_chars, max_sections=cfg.max_sections
        ))

    # Pair-level scalars, from the previous experiment's feature code unchanged.
    cos, rank = tfidf_channel(pool.rows, pool.rows)
    pair_scalars = build_raw(pool.rows, tfidf_cos=cos, tfidf_rank_pct=rank)

    # The TF-IDF encoder is fit on the pool's own text. That is a mild optimism
    # shared identically by every arm, so it cannot explain a difference between
    # them; the honest fix is per-fold fitting, which the Qwen3 path gets for
    # free since its vectors are model-derived rather than corpus-derived.
    from baselines.bert_frozen.text import candidate_to_text

    fit_texts = [candidate_to_text(p.get("matchContactFile") or {}) for p in pool.rows]
    encoder = _make_encoder(cfg, fit_texts)

    feats = build_section_features(
        pool.rows,
        list(pool.y),
        pool.seeker_ids,
        encoder=encoder,
        pair_scalars=pair_scalars,
        pair_tfidf_cos=cos,
        max_sections=cfg.max_sections,
        min_section_chars=cfg.min_section_chars,
    )
    groups = feats.groups()
    if verbose:
        print(f"rows: {feats.n_rows} over {feats.n_pairs} pairs "
              f"({feats.n_rows / max(feats.n_pairs, 1):.2f} sections/pair)")

    folds = D.seeker_disjoint_folds(feats.pair_seekers, cfg.folds, cfg.seed)
    y_pair = feats.pair_labels

    arms: dict[str, dict[str, Any]] = {}

    # ---------------------------------------------------------- logistic bar
    lg_scores = np.full(feats.n_pairs, np.nan)
    keep = np.array([r.pair_index for r in feats.rows])
    pair_first_row = {}
    for i, p in enumerate(keep):
        pair_first_row.setdefault(int(p), i)
    X_pair = feats.pair_scalars[[pair_first_row[i] for i in range(feats.n_pairs)]]

    lg_folds: list[float] = []
    for va in folds:
        tr = np.setdiff1d(np.arange(feats.n_pairs), va)
        if len(np.unique(y_pair[tr])) < 2 or len(va) == 0:
            continue
        mu, sd = X_pair[tr].mean(0), X_pair[tr].std(0)
        sd[sd < 1e-8] = 1.0
        lr = LogisticRegression(max_iter=2000, C=1.0)
        lr.fit((X_pair[tr] - mu) / sd, y_pair[tr])
        lg_scores[va] = lr.predict_proba((X_pair[va] - mu) / sd)[:, 1]
        lg_folds.append(_auc(y_pair[va], lg_scores[va]))
    arms["logistic_pair"] = {
        # Two estimators, because they disagree and the literature in this repo
        # uses both. `auc` pools every out-of-fold prediction and scores once;
        # `fold_auc_mean` averages the per-fold AUCs, which is what
        # `moe_rrf/experiment.py` reports (its published 0.6398). Mean-of-folds
        # is the noisier and more optimistic of the two at ~26 pairs per fold,
        # so the pooled figure is the one to trust — but both are recorded so
        # numbers here can be compared against either convention.
        "auc": _auc(y_pair[~np.isnan(lg_scores)], lg_scores[~np.isnan(lg_scores)]),
        "fold_auc_mean": float(np.nanmean(lg_folds)) if lg_folds else float("nan"),
        "fold_auc_std": float(np.nanstd(lg_folds)) if lg_folds else float("nan"),
        "trained_on": "12 pair scalars, seeker-disjoint CV",
    }

    # ---------------------------------------------------------- neural arms
    specs = [
        ("moe_attention", cfg.n_experts, "attention"),
        ("moe_softmax", cfg.n_experts, "softmax"),
        ("mlp_attention", 1, "attention"),
        ("moe_mean", cfg.n_experts, "mean"),
    ]

    last_gates: np.ndarray | None = None
    for name, n_exp, pooling in specs:
        arm_cfg = SectionedConfig(**{**asdict(cfg), "n_experts": n_exp, "pooling": pooling})
        scores = np.full(feats.n_pairs, np.nan)
        fold_aucs: list[float] = []
        gates_all: list[np.ndarray] = []

        for va in folds:
            tr = np.setdiff1d(np.arange(feats.n_pairs), va)
            if len(np.unique(y_pair[tr])) < 2 or len(va) == 0:
                continue
            tr_rows = D.rows_for_pairs(groups, tr)
            va_rows = D.rows_for_pairs(groups, va)

            std = RowStandardizer().fit(feats.sim[tr_rows], feats.pair_scalars[tr_rows])
            sim_tr = std.transform(feats.sim[tr_rows], feats.pair_scalars[tr_rows])
            sim_va = std.transform(feats.sim[va_rows], feats.pair_scalars[va_rows])

            res = fit(
                arm_cfg,
                sim=sim_tr,
                interaction=feats.interaction[tr_rows],
                section_emb=feats.section_emb[tr_rows],
                groups=D.regroup(groups, tr),
                pair_labels=y_pair[tr],
            )
            s, g, _ = predict(
                res.model,
                sim=sim_va,
                interaction=feats.interaction[va_rows],
                section_emb=feats.section_emb[va_rows],
                groups=D.regroup(groups, va),
            )
            scores[va] = s
            gates_all.append(g)
            fold_aucs.append(_auc(y_pair[va], s))

        ok = ~np.isnan(scores)
        arms[name] = {
            "auc": _auc(y_pair[ok], scores[ok]),
            "fold_auc_mean": float(np.nanmean(fold_aucs)) if fold_aucs else float("nan"),
            "fold_auc_std": float(np.nanstd(fold_aucs)) if fold_aucs else float("nan"),
            "n_experts": n_exp,
            "pooling": pooling,
        }
        if name == "moe_attention" and gates_all:
            last_gates = np.concatenate(gates_all)

    # ------------------------------------------------------- diagnostics
    diagnostics: dict[str, Any] | None = None
    if last_gates is not None:
        row_seekers = [feats.rows[i].seeker_id for i in range(feats.n_rows)]
        n = min(len(row_seekers), last_gates.shape[0])
        d = diag.compute(last_gates[:n, None, :], row_seekers[:n], n_permutations=300)
        diagnostics = asdict(d)
        if verbose:
            print("\n" + diag.render(d, ["accept"]))

    result: dict[str, Any] = {
        "config": {k: str(v) for k, v in asdict(cfg).items()},
        "population": {
            "pairs": feats.n_pairs,
            "rows": feats.n_rows,
            "seekers": len(set(feats.pair_seekers)),
        },
        "arms": arms,
        "diagnostics": diagnostics,
        "bar": {
            "logistic_published": cfg.holdout_bar_auc,
            "tfidf_floor": 0.5660,
            "cleared": bool(
                arms.get("moe_attention", {}).get("auc", 0) > arms["logistic_pair"]["auc"]
            ),
        },
        "holdout": None,
    }

    if verbose:
        print("\n{:<18} {:>9} {:>11} {:>9}".format(
            "arm", "AUC(OOF)", "mean-folds", "fold sd"))
        for k, v in arms.items():
            def _f(x):
                return f"{x:.4f}" if isinstance(x, float) and x == x else "—"
            print("{:<18} {:>9} {:>11} {:>9}".format(
                k, _f(v["auc"]), _f(v.get("fold_auc_mean")), _f(v.get("fold_auc_std"))))

    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Sectioned MoE experiment")
    p.add_argument("--run-id", default="sec_001")
    p.add_argument("--encoder", choices=[ENCODER_TFIDF, ENCODER_QWEN3], default=ENCODER_TFIDF)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--folds", type=int, default=None)
    p.add_argument("--max-sections", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--run-holdout",
        action="store_true",
        help="Score the frozen 69-pair holdout. Refused unless CV already beat "
        "logistic regression — the holdout is a one-shot resource.",
    )
    a = p.parse_args(argv)

    cfg = SectionedConfig(data_dir=a.data_dir, encoder=a.encoder, run_holdout=a.run_holdout)
    for field_name, val in (
        ("epochs", a.epochs), ("folds", a.folds),
        ("max_sections", a.max_sections), ("seed", a.seed),
    ):
        if val is not None:
            setattr(cfg, field_name, val)

    result = run(cfg)

    if a.run_holdout and not result["bar"]["cleared"]:
        print(
            "\nREFUSING to score the holdout: cross-validated AUC "
            f"{result['arms']['moe_attention']['auc']:.4f} did not beat logistic "
            f"regression's {result['arms']['logistic_pair']['auc']:.4f} on the same "
            "folds. The holdout is a one-shot resource — see "
            "docs/two-tower-fine-tune-plan.md."
        )

    out = cfg.output_dir / a.run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {out / 'result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
