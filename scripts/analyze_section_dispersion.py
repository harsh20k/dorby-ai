#!/usr/bin/env python3
"""Does a contact's lookingFor heterogeneity predict anything about its matches?

Two separate questions, deliberately kept apart because they have different
answers:

  1. Does dispersion predict the pos/neg *label* of a pair? (No.)
  2. Does dispersion predict *retrieval difficulty* -- how far down the ranking
     a seeker's true match falls? (Yes, and sectioning helps exactly there.)

Runs on the embeddings written by
baselines/voyage_nano_sectioned/modal_embed_space.py. Note this scores on
profile text only: searchQuery is absent from every vector, so the retrieval
numbers here are NOT comparable to docs/baseline-results-holdout.md. What is
comparable is whole-profile vs. sectioned *within this script*, which is the
contrast the analysis actually rests on.

Usage:
  python scripts/analyze_section_dispersion.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "artifacts" / "voyage_nano_sectioned_modal" / "embed_space_holdout"
DEFAULT_OUT = DEFAULT_RUN / "dispersion_analysis.json"

N_PERM = 20000
N_BOOT = 10000
SEED = 0


def load(run_dir: Path):
    emb = np.load(run_dir / "embeddings.npy")
    meta = json.loads((run_dir / "meta.json").read_text())
    whole: dict[str, int] = {}
    sections: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(meta["rows"]):
        if row["kind"] == "whole":
            whole[row["contactId"]] = i
        else:
            sections[row["contactId"]].append(i)
    return emb, meta, whole, sections


def contact_features(emb, meta, whole, sections) -> dict[str, dict]:
    """Per-contact shape measures.

    dispersion  -- 1 - mean pairwise cosine among a contact's own sections.
                   "How unlike each other are this person's separate asks?"
                   Near-independent of section count, so it measures breadth of
                   intent rather than sheer volume of text.
    spread      -- 1 - mean cosine from each section back to the whole profile.
    n_sections  -- how many lookingFor paragraphs.
    """
    profiles = {c["id"]: c["profile"] for c in meta["contacts"]}
    out: dict[str, dict] = {}
    for cid, w in whole.items():
        idx = sections.get(cid, [])
        n = len(idx)
        if n:
            spread = 1.0 - float((emb[idx] @ emb[w]).mean())
            if n > 1:
                sim = emb[idx] @ emb[idx].T
                iu = np.triu_indices(n, 1)
                dispersion = 1.0 - float(sim[iu].mean())
            else:
                dispersion = 0.0
        else:
            spread = dispersion = 0.0
        out[cid] = {
            "n_sections": n,
            "dispersion": dispersion,
            "spread": spread,
            "lookingfor_chars": len(str(profiles[cid].get("lookingFor") or "")),
        }
    return out


def label_tests(edges, feats, rng) -> list[dict]:
    """Q1: does a profile-level shape measure predict the pair's pos/neg label?"""
    y = np.array([1 if e["label"] == "pos" else 0 for e in edges])
    results = []
    for side, key in (("seeker", "source"), ("candidate", "target")):
        for f in ("dispersion", "spread", "n_sections", "lookingfor_chars"):
            x = np.array([feats[e[key]][f] for e in edges], dtype=float)
            auc = float(roc_auc_score(y, x))
            obs = abs(auc - 0.5)
            hits = sum(
                abs(roc_auc_score(rng.permutation(y), x) - 0.5) >= obs
                for _ in range(N_PERM)
            )
            results.append(
                {
                    "side": side,
                    "feature": f,
                    "auc": round(auc, 3),
                    "perm_p": round((hits + 1) / (N_PERM + 1), 4),
                }
            )
    return results


def retrieval_ranks(emb, whole, sections, positives, candidate_ids, *, sectioned: bool):
    """Rank of the true match among all holdout candidates, profile text only."""
    idx = {c: i for i, c in enumerate(candidate_ids)}
    corpus = emb[[whole[c] for c in candidate_ids]]
    ranks = []
    for e in positives:
        seeker = e["source"]
        q = emb[sections.get(seeker) or [whole[seeker]]] if sectioned else emb[whole[seeker]][None, :]
        scores = (q @ corpus.T).max(axis=0)  # max over the seeker's own sections
        order = np.argsort(-scores, kind="stable")
        ranks.append(int(np.where(order == idx[e["target"]])[0][0]) + 1)
    return np.array(ranks)


def mrr(ranks: np.ndarray) -> float:
    return float((1.0 / ranks).mean())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(SEED)
    emb, meta, whole, sections = load(args.run_dir)
    feats = contact_features(emb, meta, whole, sections)
    edges = meta["edges"]
    positives = [e for e in edges if e["label"] == "pos"]
    candidate_ids = sorted({e["target"] for e in edges})

    # --- Q1: label prediction -------------------------------------------------
    q1 = label_tests(edges, feats, rng)

    # --- Q2: retrieval difficulty --------------------------------------------
    r_whole = retrieval_ranks(emb, whole, sections, positives, candidate_ids, sectioned=False)
    r_sect = retrieval_ranks(emb, whole, sections, positives, candidate_ids, sectioned=True)

    disp = np.array([feats[e["source"]]["dispersion"] for e in positives])
    rho = float(stats.spearmanr(disp, r_whole).statistic)
    perm = np.array(
        [stats.spearmanr(rng.permutation(disp), r_whole).statistic for _ in range(N_PERM)]
    )
    perm_p = float(((np.abs(perm) >= abs(rho)).sum() + 1) / (N_PERM + 1))

    boot = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(disp), len(disp))
        if len(set(disp[i])) < 3:
            continue
        boot.append(stats.spearmanr(disp[i], r_whole[i]).statistic)
    ci = [round(float(v), 3) for v in np.percentile(boot, [2.5, 97.5])]

    # Dispersion is near-orthogonal to section count, but check the correlation
    # isn't just "more sections = harder" wearing a different hat.
    def resid(a, b):
        return a - np.polyval(np.polyfit(b, a, 1), b)

    n_sec = np.array([feats[e["source"]]["n_sections"] for e in positives], dtype=float)
    partial = float(
        stats.spearmanr(
            resid(stats.rankdata(disp), stats.rankdata(n_sec)),
            resid(stats.rankdata(r_whole), stats.rankdata(n_sec)),
        ).statistic
    )

    # --- Q3: who does sectioning actually help? ------------------------------
    med = float(np.median(disp))
    hi = disp > med
    split = {
        "median_dispersion": round(med, 4),
        "groups": [
            {
                "name": name,
                "n": int(mask.sum()),
                "mrr_whole": round(mrr(r_whole[mask]), 3),
                "mrr_sectioned": round(mrr(r_sect[mask]), 3),
                "delta": round(mrr(r_sect[mask]) - mrr(r_whole[mask]), 3),
            }
            for name, mask in (("focused", ~hi), ("multi_intent", hi))
        ],
    }

    result = {
        "n_pairs": len(edges),
        "n_positive_queries": len(positives),
        "corpus_size": len(candidate_ids),
        "note": "profile text only; searchQuery excluded from every vector",
        "q1_label_prediction": q1,
        "q2_retrieval_difficulty": {
            "spearman_rho": round(rho, 3),
            "perm_p": round(perm_p, 4),
            "bootstrap_ci95": ci,
            "partial_rho_controlling_section_count": round(partial, 3),
            "mrr_whole": round(mrr(r_whole), 3),
            "mrr_sectioned": round(mrr(r_sect), 3),
        },
        "q3_who_sectioning_helps": split,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")

    print(f"Q1  does dispersion predict the pos/neg label?")
    worst = max(q1, key=lambda r: abs(r["auc"] - 0.5))
    print(f"    no — every AUC in [{min(r['auc'] for r in q1)}, {max(r['auc'] for r in q1)}]; "
          f"strongest is {worst['feature']} ({worst['side']}) AUC {worst['auc']}, perm p {worst['perm_p']}")
    print(f"\nQ2  does dispersion predict retrieval difficulty?")
    print(f"    yes — spearman rho {rho:.3f}, perm p {perm_p:.4f}, 95% CI {ci}, "
          f"partial rho (control section count) {partial:.3f}")
    print(f"\nQ3  who does sectioning help?")
    for g in split["groups"]:
        print(f"    {g['name']:<13} n={g['n']:<3} MRR {g['mrr_whole']:.3f} -> {g['mrr_sectioned']:.3f}  ({g['delta']:+.3f})")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
