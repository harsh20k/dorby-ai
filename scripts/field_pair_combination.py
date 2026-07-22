"""Does combining multiple field-pair similarities beat the best single one?

field_pair_analysis.py found individual (seeker_field, candidate_field)
cosine-similarity scores separating labels with AUC up to ~0.61 (notes vs.
lookingFor, full n=131 sample). This asks the natural follow-up: does a
learned combination of several field-pairs do better than any single one?

Uses only field-pairs with full coverage (all 131 train pairs have both
fields present) to avoid a missing-data mess, and evaluates via 5-fold
cross-validation *within the 131 train pairs* — never touches the real
holdout, per the project's one-time-final-check rule (docs/two-tower-
fine-tune-plan.md). An in-sample fit on 131 points with many candidate
features would be meaningless without CV to catch overfitting.

Usage:
    python scripts/field_pair_combination.py --data-dir data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

FIELDS = [
    "positioning",
    "background",
    "lookingFor",
    "notes",
    "locationAvailability",
    "introPreferences",
    "personalPreferences",
    "meetingAndSchedulingPreferences",
]


def load_field_index(cache_dir: Path) -> dict[str, Any]:
    return json.loads((cache_dir / "field_index.json").read_text())


def load_vec(cache_dir: Path, cache_name: str) -> np.ndarray:
    return np.load(cache_dir / f"emb_{cache_name}.npy")[0]


def load_train_pairs(data_dir: Path) -> list[dict[str, Any]]:
    split = json.loads((data_dir / "synthetic" / "seed_split.json").read_text())
    train_pair_ids = set(split["train_pair_ids"])
    positives = json.loads((data_dir / "dataset_positive.json").read_text())
    negatives = json.loads((data_dir / "dataset_negative.json").read_text())

    pairs = []
    for label, records in (("pos", positives), ("neg", negatives)):
        for r in records:
            pid = f"{label}:{r['userContactId']}:{r['matchContactId']}"
            if pid in train_pair_ids:
                pairs.append({
                    "label": 1 if label == "pos" else 0,
                    "userContactId": r["userContactId"],
                    "matchContactId": r["matchContactId"],
                })
    return pairs


def fully_covered_field_pairs(
    pairs: list[dict[str, Any]], field_index: dict[str, Any]
) -> list[tuple[str, str]]:
    """(seeker_field, candidate_field) combos present for every single pair."""
    covered = []
    for seeker_field in FIELDS:
        for candidate_field in FIELDS:
            ok = all(
                field_index.get(p["userContactId"], {}).get(seeker_field, {}).get("present")
                and field_index.get(p["matchContactId"], {}).get(candidate_field, {}).get("present")
                for p in pairs
            )
            if ok:
                covered.append((seeker_field, candidate_field))
    return covered


def build_feature_matrix(
    pairs: list[dict[str, Any]],
    field_pairs: list[tuple[str, str]],
    field_index: dict[str, Any],
    cache_dir: Path,
) -> np.ndarray:
    X = np.zeros((len(pairs), len(field_pairs)), dtype=np.float64)
    for j, (seeker_field, candidate_field) in enumerate(field_pairs):
        for i, p in enumerate(pairs):
            s_name = field_index[p["userContactId"]][seeker_field]["cache_name"]
            c_name = field_index[p["matchContactId"]][candidate_field]["cache_name"]
            s_vec = load_vec(cache_dir, s_name)
            c_vec = load_vec(cache_dir, c_name)
            X[i, j] = float(np.dot(s_vec, c_vec))
    return X


def run(data_dir: Path, cache_dir: Path, n_folds: int = 5, seed: int = 0) -> dict[str, Any]:
    pairs = load_train_pairs(data_dir)
    field_index = load_field_index(cache_dir)
    y = np.array([p["label"] for p in pairs])

    field_pairs = fully_covered_field_pairs(pairs, field_index)
    print(f"{len(pairs)} train pairs, {len(field_pairs)} fully-covered field-pairs available as features")
    for sf, cf in field_pairs:
        print(f"  - {sf} / {cf}")

    X = build_feature_matrix(pairs, field_pairs, field_index, cache_dir)

    # Per-feature (single field-pair) AUC, for comparison.
    single_aucs = {
        f"{sf}/{cf}": roc_auc_score(y, X[:, j]) for j, (sf, cf) in enumerate(field_pairs)
    }
    best_single_name = max(single_aucs, key=lambda k: abs(single_aucs[k] - 0.5))
    best_single_auc = single_aucs[best_single_name]

    # Combined model: logistic regression, evaluated out-of-fold (5-fold CV)
    # so the 131-point fit can't just memorize noise.
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_pred = np.zeros(len(y))
    for train_idx, test_idx in skf.split(X, y):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X[train_idx], y[train_idx])
        oof_pred[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
    combined_auc = roc_auc_score(y, oof_pred)

    # Simple, less overfit-prone combiner: mean of z-scored features.
    X_z = (X - X.mean(axis=0)) / X.std(axis=0)
    mean_combo_auc = roc_auc_score(y, X_z.mean(axis=1))

    return {
        "n_pairs": len(pairs),
        "field_pairs_used": [f"{sf}/{cf}" for sf, cf in field_pairs],
        "single_aucs": single_aucs,
        "best_single": {"name": best_single_name, "auc": best_single_auc},
        "logreg_5fold_cv_auc": combined_auc,
        "mean_zscore_combo_auc": mean_combo_auc,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/field_embeddings/nano"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/field_embeddings/field_pair_combination.json"))
    args = parser.parse_args()

    result = run(args.data_dir, args.cache_dir)

    print("\nSingle field-pair AUCs (fully-covered only):")
    for name, auc in sorted(result["single_aucs"].items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {name:45s} {auc:.4f}")

    print(f"\nBest single field-pair: {result['best_single']['name']} (AUC={result['best_single']['auc']:.4f})")
    print(f"Combined (logistic regression, 5-fold CV out-of-fold AUC): {result['logreg_5fold_cv_auc']:.4f}")
    print(f"Combined (simple mean of z-scored features, in-sample):    {result['mean_zscore_combo_auc']:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
