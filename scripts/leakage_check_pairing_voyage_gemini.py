"""Basic leakage/circularity checks on a pairing_voyage_gemini batch.

Read-only diagnostic, no training. Reuses the same three probes this project
has run on every prior synthetic batch before trusting it as training data:

1. Candidate-profile-only leakage: can a plain TF-IDF+LogReg classifier guess
   the label from the candidate's own profile text alone, with no query and
   no seeker text? (This is exactly the mechanism that made batch_500_001's
   label 99.2%-predictable and wrecked twotower run_001 — possible-bugs.md #4.)
2. Per-seeker base rate: what fraction of seekers are all-positive or
   all-negative across their queries, and how well does seeker identity alone
   (via leave-one-out positive rate) predict the label?
3. Lexical circularity: does plain TF-IDF cosine between searchQuery and
   candidate text already predict the label (the same check that flagged
   pair_test_001's 0.868 AUC and rrf_002's 0.701 AUC)?

Usage:
    python scripts/leakage_check_pairing_voyage_gemini.py \
        --batch-dir artifacts/pairing_voyage_gemini/smoke_test_002
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

PROFILE_FIELDS = (
    "positioning",
    "background",
    "lookingFor",
    "notes",
    "locationAvailability",
    "introPreferences",
    "personalPreferences",
    "meetingAndSchedulingPreferences",
)


def profile_to_text(profile: dict) -> str:
    parts = []
    for field in PROFILE_FIELDS:
        value = profile.get(field)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return "\n\n".join(parts)


def load_batch(batch_dir: Path) -> list[dict]:
    manifest = json.loads((batch_dir / "manifest.json").read_text())
    records = []
    for rec in manifest["records"]:
        path = batch_dir / rec["path"]
        payload = json.loads(path.read_text())
        records.append(
            {
                "seeker_id": rec["seeker_id"],
                "candidate_id": rec["candidate_id"],
                "label": 1 if payload["label"] == "pos" else 0,
                "query": payload["pair"].get("searchQuery", "") or "",
                "candidate_text": profile_to_text(payload["pair"]["matchContactFile"]),
            }
        )
    return records


def cv_auc(x_text: list[str], y: np.ndarray, n_splits: int = 5) -> float:
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)
    x = vec.fit_transform(x_text)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    probs = cross_val_predict(clf, x, y, cv=skf, method="predict_proba")[:, 1]
    return roc_auc_score(y, probs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", type=Path, required=True)
    args = parser.parse_args()

    records = load_batch(args.batch_dir)
    n = len(records)
    y = np.array([r["label"] for r in records])
    print(f"Loaded {n} pairs ({y.sum()} pos / {n - y.sum()} neg) from {args.batch_dir}")

    # 1. Candidate-only leakage
    cand_text = [r["candidate_text"] for r in records]
    cand_auc = cv_auc(cand_text, y)
    print(f"\n[1] Candidate-profile-only AUC (5-fold CV): {cand_auc:.4f}")

    # 2. Per-seeker base rate
    by_seeker: dict[str, list[int]] = defaultdict(list)
    for r in records:
        by_seeker[r["seeker_id"]].append(r["label"])
    n_seekers = len(by_seeker)
    all_pos = sum(1 for labels in by_seeker.values() if all(l == 1 for l in labels))
    all_neg = sum(1 for labels in by_seeker.values() if all(l == 0 for l in labels))
    both = n_seekers - all_pos - all_neg
    print(f"\n[2] Per-seeker base rate: {n_seekers} seekers")
    print(f"    all-positive: {all_pos} ({all_pos / n_seekers:.1%})")
    print(f"    all-negative: {all_neg} ({all_neg / n_seekers:.1%})")
    print(f"    both classes: {both} ({both / n_seekers:.1%})")

    # leave-one-out seeker positive rate as sole predictor
    seeker_ids = np.array([r["seeker_id"] for r in records])
    loo_score = np.zeros(n)
    for seeker, labels in by_seeker.items():
        idx = np.where(seeker_ids == seeker)[0]
        total = len(labels)
        s = sum(labels)
        for i in idx:
            lbl = y[i]
            denom = total - 1
            loo_score[i] = (s - lbl) / denom if denom > 0 else 0.5
    seeker_auc = roc_auc_score(y, loo_score)
    print(f"    seeker-identity-only AUC (leave-one-out positive rate): {seeker_auc:.4f}")

    # 3. Lexical circularity: TF-IDF cosine(query, candidate) vs label
    query_text = [r["query"] for r in records]
    combined_vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 1), min_df=2)
    all_text = query_text + cand_text
    combined_vec.fit(all_text)
    q_vecs = combined_vec.transform(query_text)
    c_vecs = combined_vec.transform(cand_text)
    q_norm = q_vecs.multiply(1.0 / (np.sqrt(q_vecs.multiply(q_vecs).sum(axis=1)) + 1e-9))
    c_norm = c_vecs.multiply(1.0 / (np.sqrt(c_vecs.multiply(c_vecs).sum(axis=1)) + 1e-9))
    cosine = np.asarray(q_norm.multiply(c_norm).sum(axis=1)).ravel()
    lexical_auc = roc_auc_score(y, cosine)
    print(f"\n[3] Lexical circularity — TF-IDF query-candidate cosine AUC: {lexical_auc:.4f}")

    print("\n--- summary ---")
    print(f"candidate-only:    {cand_auc:.4f}")
    print(f"seeker-identity:   {seeker_auc:.4f}  ({both}/{n_seekers} seekers carry both classes)")
    print(f"lexical (TF-IDF):  {lexical_auc:.4f}")


if __name__ == "__main__":
    main()
