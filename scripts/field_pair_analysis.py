"""Which seeker-field / candidate-field pairing actually predicts a good intro?

Uses the cached per-field voyage-4-nano embeddings (scripts/embed_fields_
experiment.py, artifacts/field_embeddings/nano/field_index.json) for the 131
frozen train pairs (71 pos / 60 neg, non-holdout, see docs/possible-bugs.md
#3 / seed_split.json). For every (seeker_field, candidate_field) combo out
of 8x8=64, computes cosine similarity per pair and measures how well that
single number separates positive from negative pairs (ROC-AUC) — same
metric used for every other baseline, so results are directly comparable
to the TF-IDF/BERT/Voyage floor (0.59) and the earlier KG-hub-attribute
experiment (0.50, chance).

This tells us which specific parts of a profile actually carry matching
signal, rather than assuming (as every existing baseline does) that the
whole blob matters equally.

Usage:
    python scripts/field_pair_analysis.py --data-dir data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

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
                pairs.append({"label": label, "userContactId": r["userContactId"], "matchContactId": r["matchContactId"]})
    return pairs


def run(data_dir: Path, cache_dir: Path, min_n: int = 20) -> list[dict[str, Any]]:
    pairs = load_train_pairs(data_dir)
    n_pos = sum(1 for p in pairs if p["label"] == "pos")
    n_neg = sum(1 for p in pairs if p["label"] == "neg")
    print(f"train pairs: {len(pairs)} ({n_pos} pos / {n_neg} neg)")

    field_index = load_field_index(cache_dir)

    results = []
    for seeker_field in FIELDS:
        for candidate_field in FIELDS:
            pos_scores, neg_scores = [], []
            for p in pairs:
                seeker_entry = field_index.get(p["userContactId"], {}).get(seeker_field, {})
                cand_entry = field_index.get(p["matchContactId"], {}).get(candidate_field, {})
                if not seeker_entry.get("present") or not cand_entry.get("present"):
                    continue
                s_vec = load_vec(cache_dir, seeker_entry["cache_name"])
                c_vec = load_vec(cache_dir, cand_entry["cache_name"])
                sim = float(np.dot(s_vec, c_vec))
                (pos_scores if p["label"] == "pos" else neg_scores).append(sim)

            n = len(pos_scores) + len(neg_scores)
            if n < min_n or not pos_scores or not neg_scores:
                continue

            y_true = [1] * len(pos_scores) + [0] * len(neg_scores)
            y_score = pos_scores + neg_scores
            auc = roc_auc_score(y_true, y_score)

            results.append({
                "seeker_field": seeker_field,
                "candidate_field": candidate_field,
                "n_pairs": n,
                "n_pos": len(pos_scores),
                "n_neg": len(neg_scores),
                "auc": auc,
                "mean_sim_pos": float(np.mean(pos_scores)),
                "mean_sim_neg": float(np.mean(neg_scores)),
            })

    results.sort(key=lambda r: abs(r["auc"] - 0.5), reverse=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/field_embeddings/nano"))
    parser.add_argument("--min-n", type=int, default=20, help="minimum pairs with both fields present")
    parser.add_argument("--output", type=Path, default=Path("artifacts/field_embeddings/field_pair_auc.json"))
    args = parser.parse_args()

    results = run(args.data_dir, args.cache_dir, args.min_n)

    print(f"\n{len(results)} field-pairs with >= {args.min_n} labeled examples, ranked by |AUC - 0.5|:\n")
    print(f"{'seeker_field':32s} {'candidate_field':32s} {'n':>5s} {'AUC':>7s} {'pos_sim':>8s} {'neg_sim':>8s}")
    for r in results:
        print(
            f"{r['seeker_field']:32s} {r['candidate_field']:32s} {r['n_pairs']:5d} "
            f"{r['auc']:7.4f} {r['mean_sim_pos']:8.4f} {r['mean_sim_neg']:8.4f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
