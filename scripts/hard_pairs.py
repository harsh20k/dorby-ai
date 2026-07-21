"""Rank the original real pairs by how badly Voyage got them wrong.

Reconstructs per-pair cosine scores for the frozen Voyage baselines *entirely
from the on-disk embedding caches* (zero API calls), then ranks every pair by
its signed margin to that model's best-F1 decision threshold. A negative margin
means the pair is misclassified at the operating point the eval reports; the
most-negative margins are the "hardest" pairs (hard positives that should match
but score low, hard negatives that shouldn't match but score high).

Scope: the *original* dataset — the 100 positive / 100 negative real seed pairs
(``matchContactId`` / ``userContactId`` without the ``cmsynth`` prefix). Those
are exactly the pairs the full-dataset ``voyage_large`` / ``voyage_nano`` caches
were built on; synthetic pairs promoted later are not in those caches.

Scoring matches ``baselines/voyage_*/eval.py`` exactly:
  score = cosine(seeker_emb, candidate_emb)   (embeddings are L2-normalized)
  seeker text     = seeker_to_text(userContactFile, searchQuery)   [input_type=query]
  candidate text  = candidate_to_text(matchContactFile)            [input_type=document]
  threshold t     = pair.best_f1_threshold from that model's metrics.json

  positive pair margin = score - t   (predicted positive iff score >= t)
  negative pair margin = t - score
  misclassified        = margin < 0

Run from the repo root (needs the local ``data/`` and ``artifacts/`` dirs, both
gitignored, so this reads them from wherever cwd points):

  .venv/bin/python scripts/hard_pairs.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from baselines.bert_frozen.text import candidate_to_text, seeker_to_text


def is_synth(record: dict[str, Any]) -> bool:
    return record["matchContactId"].startswith("cmsynth") or record[
        "userContactId"
    ].startswith("cmsynth")


def load_real_pairs(data_dir: Path) -> tuple[list[dict], list[dict]]:
    pos = json.loads((data_dir / "dataset_positive.json").read_text())
    neg = json.loads((data_dir / "dataset_negative.json").read_text())
    pos = [r for r in pos if not is_synth(r)]
    neg = [r for r in neg if not is_synth(r)]
    return pos, neg


def large_cache_key(text: str, model_name: str, input_type: str, dim: int) -> str:
    """Mirror baselines/voyage_large/encode.py::text_cache_key exactly."""
    h = hashlib.sha256()
    h.update(model_name.encode())
    h.update(b"\0")
    h.update(input_type.encode())
    h.update(b"\0")
    h.update(str(dim).encode())
    h.update(b"\0")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def scores_from_large_cache(
    pos: list[dict], neg: list[dict], artifacts_dir: Path, meta: dict
) -> tuple[np.ndarray, np.ndarray, int]:
    emb_dir = artifacts_dir / "emb"
    model_name = meta["model_name"]
    dim = meta["output_dimension"]
    misses = 0

    def load(text: str, input_type: str) -> np.ndarray | None:
        nonlocal misses
        p = emb_dir / f"{large_cache_key(text, model_name, input_type, dim)}.npy"
        if not p.exists():
            misses += 1
            return None
        return l2(np.load(p).reshape(-1).astype(np.float32))

    def pair_scores(records: list[dict]) -> np.ndarray:
        out = []
        for r in records:
            s = load(seeker_to_text(r["userContactFile"], r["searchQuery"]), "query")
            c = load(candidate_to_text(r["matchContactFile"]), "document")
            out.append(float(np.dot(s, c)) if s is not None and c is not None else np.nan)
        return np.array(out, dtype=np.float64)

    return pair_scores(pos), pair_scores(neg), misses


def scores_from_nano_cache(
    pos: list[dict], neg: list[dict], artifacts_dir: Path
) -> tuple[np.ndarray, np.ndarray, int]:
    """Nano stores positional grouped arrays aligned to the original pair order.

    Verified: the first 100 positives / 100 negatives in the current dataset are
    exactly the original real seed, in the order the cache was built.
    """
    ps = l2_rows(np.load(artifacts_dir / "emb_pos_seeker.npy"))
    pc = l2_rows(np.load(artifacts_dir / "emb_pos_cand.npy"))
    ns = l2_rows(np.load(artifacts_dir / "emb_neg_seeker.npy"))
    nc = l2_rows(np.load(artifacts_dir / "emb_neg_cand.npy"))
    if not (len(ps) >= len(pos) and len(ns) >= len(neg)):
        raise SystemExit(
            f"nano cache too small: pos_emb={len(ps)} vs {len(pos)} pairs, "
            f"neg_emb={len(ns)} vs {len(neg)} pairs"
        )
    pos_scores = np.sum(ps[: len(pos)] * pc[: len(pos)], axis=1).astype(np.float64)
    neg_scores = np.sum(ns[: len(neg)] * nc[: len(neg)], axis=1).astype(np.float64)
    return pos_scores, neg_scores, 0


def l2_rows(m: np.ndarray) -> np.ndarray:
    m = m.astype(np.float32)
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.clip(n, 1e-12, None)


def build_rows(
    model: str,
    pos: list[dict],
    neg: list[dict],
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
    threshold: float,
) -> list[dict]:
    rows = []
    for label, records, scores in (("pos", pos, pos_scores), ("neg", neg, neg_scores)):
        for r, s in zip(records, scores):
            if np.isnan(s):
                margin = np.nan
                mis = None
            elif label == "pos":
                margin = s - threshold
                mis = bool(margin < 0)
            else:
                margin = threshold - s
                mis = bool(margin < 0)
            rows.append(
                {
                    "model": model,
                    "label": label,
                    "userContactId": r["userContactId"],
                    "matchContactId": r["matchContactId"],
                    "searchQuery": (r.get("searchQuery") or "").replace("\n", " ")[:160],
                    "score": None if np.isnan(s) else round(float(s), 6),
                    "threshold": round(float(threshold), 6),
                    "margin": None if np.isnan(margin) else round(float(margin), 6),
                    "misclassified": mis,
                }
            )
    # Hardest first: most-negative margin. NaN (uncached) sorted last.
    rows.sort(key=lambda x: (x["margin"] is None, x["margin"] if x["margin"] is not None else 0.0))
    for i, row in enumerate(rows, 1):
        row["rank"] = i
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/hard_pairs"))
    args = ap.parse_args()

    pos, neg = load_real_pairs(args.data_dir)
    print(f"original real pairs: {len(pos)} positive / {len(neg)} negative")

    large_meta = json.loads((args.artifacts_root / "voyage_large" / "metrics.json").read_text())
    nano_meta = json.loads((args.artifacts_root / "voyage_nano" / "metrics.json").read_text())
    large_t = large_meta["pair"]["best_f1_threshold"]
    nano_t = nano_meta["pair"]["best_f1_threshold"]

    lps, lns, lmiss = scores_from_large_cache(
        pos, neg, args.artifacts_root / "voyage_large", large_meta
    )
    nps, nns, nmiss = scores_from_nano_cache(pos, neg, args.artifacts_root / "voyage_nano")
    print(f"voyage-4-large: cache misses={lmiss}  threshold={large_t:.4f}")
    print(f"voyage-4-nano:  cache misses={nmiss}  threshold={nano_t:.4f}")

    large_rows = build_rows("voyage-4-large", pos, neg, lps, lns, large_t)
    nano_rows = build_rows("voyage-4-nano", pos, neg, nps, nns, nano_t)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank", "model", "label", "userContactId", "matchContactId",
        "searchQuery", "score", "threshold", "margin", "misclassified",
    ]
    for name, rows in (("large", large_rows), ("nano", nano_rows)):
        csv_path = args.out_dir / f"hard_pairs_{name}.csv"
        with csv_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for row in rows:
                w.writerow({k: row.get(k) for k in fields})
        (args.out_dir / f"hard_pairs_{name}.json").write_text(json.dumps(rows, indent=2) + "\n")
        n_mis = sum(1 for r in rows if r["misclassified"])
        print(f"{name}: wrote {csv_path.name} + .json  ({n_mis}/{len(rows)} misclassified @ best-F1)")

    # Consensus view: pairs both models get wrong, by combined margin.
    key = lambda r: (r["label"], r["userContactId"], r["matchContactId"])
    nano_by = {key(r): r for r in nano_rows}
    both = []
    for lr in large_rows:
        nr = nano_by.get(key(lr))
        if nr is None or lr["margin"] is None or nr["margin"] is None:
            continue
        both.append(
            {
                "label": lr["label"],
                "userContactId": lr["userContactId"],
                "matchContactId": lr["matchContactId"],
                "searchQuery": lr["searchQuery"],
                "large_score": lr["score"],
                "large_margin": lr["margin"],
                "nano_score": nr["score"],
                "nano_margin": nr["margin"],
                "both_wrong": bool(lr["misclassified"] and nr["misclassified"]),
                "sum_margin": round(lr["margin"] + nr["margin"], 6),
            }
        )
    both.sort(key=lambda x: x["sum_margin"])
    (args.out_dir / "hard_pairs_consensus.json").write_text(json.dumps(both, indent=2) + "\n")
    n_both = sum(1 for r in both if r["both_wrong"])
    print(f"consensus: wrote hard_pairs_consensus.json  ({n_both} pairs wrong by BOTH models)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
