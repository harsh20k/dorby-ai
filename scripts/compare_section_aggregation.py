#!/usr/bin/env python3
"""Compare section-aggregation shapes on the real holdout, from one embedding pass.

The question this answers: when a seeker has several separate ``lookingFor``
asks, is a candidate good because it *fits one ask well* (relevance-shaped
aggregation: max / softmax) or bad because it *fails one ask badly*
(veto-shaped: min / softmin / noisy_or)?

That matters before choosing a mixture-of-experts combine rule. Averaging
independent sub-scores is what lost as ``structured_cot``
(``docs/llm-judge-experiment.md``); a veto does not average. If veto-shaped
aggregation beats relevance-shaped here, the dealbreaker theory has support and
a soft-min combine is worth building. If it does not, the combine rule should
stay relevance-shaped.

Isolation: this reads shared baseline helpers but modifies none of them. The
aggregation shapes live in ``moe_reranker/aggregation.py`` and the scoring loop
in ``moe_reranker/section_scoring.py``, both owned by this experiment.

Free to run: every mode is scored from the *same* cached embedding pass, so
nothing re-encodes. Point ``--artifacts-dir`` at a directory that already holds
the cached section/corpus embeddings and pass the ``--max-length`` they were
built with (it is part of the cache key).

    .venv/bin/python scripts/compare_section_aggregation.py \
      --artifacts-dir artifacts/voyage_nano_sectioned_seeker_softmax_local \
      --max-length 4096 --holdout-only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from moe_reranker.aggregation import AGG_FAMILY
from moe_reranker.section_scoring import score_all_aggregations

# Temperature sweeps are cheap (pure re-aggregation of a cached matrix), so both
# soft modes get a small sweep rather than a single guessed value. T=0.05 is the
# value already documented for softmax, kept so the table anchors to a known row.
AGG_CONFIGS: list[dict[str, Any]] = [
    # relevance-shaped: "does any ask fit?"
    {"label": "max", "agg": "max"},
    {"label": "topk_mean(k=2)", "agg": "topk_mean", "topk": 2},
    {"label": "softmax(T=0.01)", "agg": "softmax", "temperature": 0.01},
    {"label": "softmax(T=0.05)", "agg": "softmax", "temperature": 0.05},
    {"label": "softmax(T=0.20)", "agg": "softmax", "temperature": 0.20},
    # plain average: the control, i.e. the shape that lost as structured_cot
    {"label": "mean", "agg": "mean"},
    # veto-shaped: "is any ask badly violated?"
    {"label": "min", "agg": "min"},
    {"label": "softmin(T=0.01)", "agg": "softmin", "temperature": 0.01},
    {"label": "softmin(T=0.05)", "agg": "softmin", "temperature": 0.05},
    {"label": "softmin(T=0.20)", "agg": "softmin", "temperature": 0.20},
    {"label": "noisy_or", "agg": "noisy_or"},
]

ROW_KEYS = ("pair_auc", "avg_precision", "mrr", "top1", "recall_at_10", "hard_neg_auc")


def _extract(metrics: dict[str, Any]) -> dict[str, float | None]:
    pair = metrics["pair"]
    retr = metrics["retrieval"]
    slices = metrics.get("slices") or {}

    # neg_hardness splits negatives by seeker/candidate token overlap; the hard
    # slice is the only population that exists in production, so it is the row
    # that actually matters here. Key is "pair_auc", not "roc_auc".
    hard = None
    blob = (slices.get("neg_hardness") or {}).get("hard")
    if isinstance(blob, dict) and blob.get("pair_auc") is not None:
        hard = float(blob["pair_auc"])

    return {
        "pair_auc": float(pair["roc_auc"]),
        "avg_precision": float(pair["average_precision"]),
        "mrr": float(retr["mrr"]),
        "top1": float(retr.get("top1", retr.get("recall@1"))),
        "recall_at_10": float(retr["recall@10"]),
        "hard_neg_auc": hard,
    }


def _fmt(v: float | None) -> str:
    return "  n/a " if v is None else f"{v:.4f}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--model", type=str, default="voyageai/voyage-4-nano")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument(
        "--max-length",
        type=int,
        default=4096,
        help="Must match the cached embeddings' max_length (it is in the cache key)",
    )
    p.add_argument("--truncate-dim", type=int, default=1024)
    p.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts/voyage_nano_sectioned_seeker_softmax_local"),
    )
    p.add_argument("--holdout-only", action="store_true", default=True)
    p.add_argument("--split-path", type=Path, default=None)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/section_aggregation_comparison.json"),
    )
    args = p.parse_args()

    results = score_all_aggregations(
        data_dir=args.data_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        truncate_dim=args.truncate_dim,
        artifacts_dir=args.artifacts_dir,
        agg_configs=AGG_CONFIGS,
        holdout_only=args.holdout_only,
        split_path=args.split_path,
    )

    rows = {label: _extract(m) for label, m in results.items()}

    header = (
        f"{'mode':<18} {'family':<10} {'pairAUC':>8} {'AP':>8} "
        f"{'MRR':>8} {'top1':>8} {'R@10':>8} {'hardAUC':>8}"
    )
    print("\n" + header)
    print("-" * len(header))
    last_family = None
    for cfg in AGG_CONFIGS:
        label = cfg["label"]
        fam = AGG_FAMILY[cfg["agg"]]
        if last_family is not None and fam != last_family:
            print("-" * len(header))
        last_family = fam
        r = rows[label]
        print(
            f"{label:<18} {fam:<10} "
            f"{_fmt(r['pair_auc']):>8} {_fmt(r['avg_precision']):>8} "
            f"{_fmt(r['mrr']):>8} {_fmt(r['top1']):>8} "
            f"{_fmt(r['recall_at_10']):>8} {_fmt(r['hard_neg_auc']):>8}"
        )

    # Verdict on the question the script exists to answer.
    def best_of(family: str, key: str) -> tuple[str, float] | None:
        cands = [
            (cfg["label"], rows[cfg["label"]][key])
            for cfg in AGG_CONFIGS
            if AGG_FAMILY[cfg["agg"]] == family and rows[cfg["label"]][key] is not None
        ]
        return max(cands, key=lambda t: t[1]) if cands else None

    print()
    verdict: dict[str, Any] = {}
    for key in ("pair_auc", "mrr", "hard_neg_auc"):
        rel, veto = best_of("relevance", key), best_of("veto", key)
        if rel is None or veto is None:
            print(f"{key:<14} not available in this run")
            continue
        (rel_label, rel_v), (veto_label, veto_v) = rel, veto
        winner = "veto" if veto_v > rel_v else "relevance"
        verdict[key] = {
            "best_relevance": {"mode": rel_label, "value": rel_v},
            "best_veto": {"mode": veto_label, "value": veto_v},
            "winner": winner,
            "delta_veto_minus_relevance": veto_v - rel_v,
        }
        print(
            f"{key:<14} best relevance {rel_label} {rel_v:.4f} | "
            f"best veto {veto_label} {veto_v:.4f} | "
            f"winner: {winner} ({veto_v - rel_v:+.4f})"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "model": args.model,
                "max_length": args.max_length,
                "truncate_dim": args.truncate_dim,
                "holdout_only": args.holdout_only,
                "n_positives": results[AGG_CONFIGS[0]["label"]]["pair"]["num_positive"],
                "n_negatives": results[AGG_CONFIGS[0]["label"]]["pair"]["num_negative"],
                "rows": rows,
                "verdict": verdict,
                "full": results,
            },
            indent=2,
        )
    )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
