#!/usr/bin/env python3
"""Quantify how "cheatable" candidate-only text is for pos/neg classification.

A real matcher needs the seeker's profile + searchQuery to decide if a
candidate is a good intro. This script deliberately withholds that and asks:
can a trivial bag-of-words classifier guess the label from the CANDIDATE
profile text alone? If yes, well above chance, the text carries a label
shortcut (e.g. generator-prompt style tells) rather than requiring the
seeker/candidate relationship to be understood.

Run on real-only, synthetic-only, and combined populations from the
canonical dataset, or against an unpromoted batch's staged pairs directly
(useful for validating a generation-prompt fix before promotion).

Usage:
    python3 scripts/check_synth_cheatability.py
    python3 scripts/check_synth_cheatability.py --batch-dir artifacts/synth/batch_pilot_fix_001 --folds 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.bert_frozen.text import candidate_to_text  # noqa: E402


def _is_synth(pair: dict[str, Any]) -> bool:
    return str(pair["userContactId"]).startswith("cmsynth") or str(
        pair["matchContactId"]
    ).startswith("cmsynth")


def load_canonical(data_dir: Path) -> tuple[list[dict], list[dict]]:
    pos = json.loads((data_dir / "dataset_positive.json").read_text())
    neg = json.loads((data_dir / "dataset_negative.json").read_text())
    return pos, neg


def load_batch_staged(batch_dir: Path) -> tuple[list[dict], list[dict]]:
    """Load candidate-bearing pairs from an unpromoted batch's staged/ dir."""
    pos: list[dict] = []
    neg: list[dict] = []
    for path in sorted((batch_dir / "staged").glob("*.json")):
        env = json.loads(path.read_text())
        pair = env.get("pair")
        if not pair:
            continue
        (pos if env.get("label") == "pos" else neg).append(pair)
    return pos, neg


def candidate_texts(pairs: list[dict]) -> list[str]:
    return [candidate_to_text(p["matchContactFile"]) for p in pairs]


def cheatability_auc(
    pos_texts: list[str], neg_texts: list[str], *, folds: int, seed: int
) -> dict[str, Any]:
    texts = pos_texts + neg_texts
    labels = np.array([1] * len(pos_texts) + [0] * len(neg_texts))
    n_min_class = min(len(pos_texts), len(neg_texts))
    if n_min_class < 2:
        return {"n_pos": len(pos_texts), "n_neg": len(neg_texts), "auc_mean": None, "auc_folds": None, "note": "too few examples"}

    k = max(2, min(folds, n_min_class))
    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=2000, ngram_range=(1, 2), min_df=1)),
            ("clf", LogisticRegression(max_iter=2000)),
        ]
    )
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    scores = cross_val_score(pipe, texts, labels, cv=skf, scoring="roc_auc")
    return {
        "n_pos": len(pos_texts),
        "n_neg": len(neg_texts),
        "folds": k,
        "auc_mean": float(np.mean(scores)),
        "auc_std": float(np.std(scores)),
        "auc_folds": [float(s) for s in scores],
    }


def run(data_dir: Path, batch_dir: Path | None, folds: int, seed: int) -> dict[str, Any]:
    pos, neg = load_canonical(data_dir)
    real_pos = [p for p in pos if not _is_synth(p)]
    real_neg = [p for p in neg if not _is_synth(p)]
    synth_pos = [p for p in pos if _is_synth(p)]
    synth_neg = [p for p in neg if _is_synth(p)]

    results: dict[str, Any] = {}
    results["real_only"] = cheatability_auc(
        candidate_texts(real_pos), candidate_texts(real_neg), folds=folds, seed=seed
    )
    results["synthetic_only"] = cheatability_auc(
        candidate_texts(synth_pos), candidate_texts(synth_neg), folds=folds, seed=seed
    )
    results["combined"] = cheatability_auc(
        candidate_texts(real_pos + synth_pos),
        candidate_texts(real_neg + synth_neg),
        folds=folds,
        seed=seed,
    )

    if batch_dir is not None:
        batch_pos, batch_neg = load_batch_staged(batch_dir)
        results[f"batch:{batch_dir.name}"] = cheatability_auc(
            candidate_texts(batch_pos), candidate_texts(batch_neg), folds=folds, seed=seed
        )

    if results["real_only"]["auc_mean"] is not None and results["synthetic_only"]["auc_mean"] is not None:
        results["gap_synthetic_minus_real"] = (
            results["synthetic_only"]["auc_mean"] - results["real_only"]["auc_mean"]
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--batch-dir", type=Path, default=None, help="e.g. artifacts/synth/batch_pilot_fix_001")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = run(args.data_dir, args.batch_dir, args.folds, args.seed)

    print("\n=== Candidate-only cheatability (ROC-AUC, no seeker/query seen) ===\n")
    for name, r in results.items():
        if name == "gap_synthetic_minus_real":
            continue
        if r.get("auc_mean") is None:
            print(f"{name:20s} n_pos={r['n_pos']:>4d} n_neg={r['n_neg']:>4d}  {r.get('note', '')}")
            continue
        print(
            f"{name:20s} n_pos={r['n_pos']:>4d} n_neg={r['n_neg']:>4d} "
            f"folds={r['folds']}  AUC = {r['auc_mean']:.4f} (+/- {r['auc_std']:.4f})"
        )
    if "gap_synthetic_minus_real" in results:
        print(f"\n{'gap (synthetic - real)':20s} {results['gap_synthetic_minus_real']:+.4f}")
        print(
            "(0.5 = pure chance; a large positive gap means synthetic candidate "
            "text alone reveals the label in a way real text doesn't — the "
            "diagnostic this script exists to catch)"
        )

    out_path = ROOT / "artifacts" / "synth_cheatability.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
