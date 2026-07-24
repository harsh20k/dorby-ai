#!/usr/bin/env python3
"""Sanity-check the scorer that labels pairing batches.

`synth_pipeline/pairing/label.py` fits the TF-IDF+Voyage-nano fusion on real
train pairs and then uses it to assign every synthetic label. If that fit is
wrong, every label in every batch is wrong with it — and nothing else in the
pipeline would notice, because there is no judge and no human in the loop.

So: fit exactly the way pairing does, score the frozen 69-pair real holdout,
and compare against the number `docs/baseline-results-holdout.md` records for
the same scorer (pair ROC-AUC 0.6397). A close match means the scorer used for
labeling is the one that was actually measured.

Run this AFTER building a batch — it's a verification add-on, not a gate.

    python scripts/verify_pairing_scorer.py
    python scripts/verify_pairing_scorer.py --data-dir /path/to/data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.bert_frozen.text import candidate_to_text, seeker_to_text  # noqa: E402
from baselines.holdout import filter_to_holdout  # noqa: E402
from baselines.metrics import pair_metrics  # noqa: E402
from synth_pipeline.pairing.label import compute_thresholds, fit_scorer  # noqa: E402

# docs/baseline-results-holdout.md, "Hybrid TF-IDF+nano" column.
EXPECTED_AUC = 0.6397
TOLERANCE = 0.05


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--split-path", type=Path, default=None)
    ap.add_argument("--cache-dir", type=Path, default=ROOT / "artifacts" / "pairing" / "_verify_cache")
    ap.add_argument("--fusion-mode", choices=("alpha", "logistic"), default="alpha")
    ap.add_argument("--tolerance", type=float, default=TOLERANCE)
    args = ap.parse_args()

    split_path = args.split_path or (args.data_dir / "synthetic" / "seed_split.json")

    print("fitting the pairing scorer (real train pairs only)...")
    scorer = fit_scorer(
        args.data_dir, split_path,
        cache_dir=args.cache_dir, fusion_mode=args.fusion_mode,
    )
    thresholds = compute_thresholds(scorer)
    print(f"  fit on {scorer.n_fit} real pairs | fit AUC={scorer.fusion.fit_auc:.4f}")
    print(f"  deadband: <={thresholds.lower:.4f} neg | >={thresholds.upper:.4f} pos")

    positives = json.loads((args.data_dir / "dataset_positive.json").read_text(encoding="utf-8"))
    negatives = json.loads((args.data_dir / "dataset_negative.json").read_text(encoding="utf-8"))
    positives, negatives = filter_to_holdout(positives, negatives, split_path)
    print(f"holdout: {len(positives)} pos / {len(negatives)} neg")

    pos_scores = scorer.score(
        [seeker_to_text(r["userContactFile"], r["searchQuery"]) for r in positives],
        [candidate_to_text(r["matchContactFile"]) for r in positives],
        cache_prefix="verify_hold_pos",
    )
    neg_scores = scorer.score(
        [seeker_to_text(r["userContactFile"], r["searchQuery"]) for r in negatives],
        [candidate_to_text(r["matchContactFile"]) for r in negatives],
        cache_prefix="verify_hold_neg",
    )

    pair = pair_metrics(pos_scores, neg_scores)
    auc = pair["roc_auc"]
    delta = auc - EXPECTED_AUC
    print()
    print(f"holdout pair ROC-AUC : {auc:.4f}")
    print(f"documented (hybrid)  : {EXPECTED_AUC:.4f}")
    print(f"delta                : {delta:+.4f}")
    print(f"best F1              : {pair['best_f1']:.4f}")

    # How the deadband would split this known-truth population — a direct read on
    # how much of the holdout the labeler would even venture an opinion about.
    import numpy as np

    all_scores = np.concatenate([pos_scores, neg_scores])
    labeled = int(((all_scores >= thresholds.upper) | (all_scores <= thresholds.lower)).sum())
    print(f"deadband would label : {labeled}/{len(all_scores)} holdout pairs "
          f"({100*labeled/len(all_scores):.0f}%)")

    ok = abs(delta) <= args.tolerance
    print()
    print("PASS — labeling scorer matches its documented performance." if ok
          else f"FAIL — off by more than {args.tolerance}; labels in existing batches are suspect.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
