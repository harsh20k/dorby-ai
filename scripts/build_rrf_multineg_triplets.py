"""Extract (anchor, positive, negative_1..negative_k) rows from a pairing_rrf batch.

Sibling to scripts/build_rrf_triplets.py (not imported from it, by design —
this feeds the isolated twotower_rrf_triplet_bigbatch/ experiment, which must
not depend on any other experiment's code). Same query_key grouping and same
seeker_to_text/candidate_to_text serialization, but instead of the cartesian
pos x neg expansion (one row per single negative), each row samples up to
`--negatives-per-anchor` unique negatives from that query_key's negative pool
so a single training row can contrast one positive against several negatives
at once.

These labels are an LLM judge's opinion on synthetic profiles, not real
accept/decline outcomes — this output is training data for an experimental
arm, not something promoted into data/dataset_*.json.

Usage:
  # first inspect real per-query_key negative coverage to pick k
  python scripts/build_rrf_multineg_triplets.py --batch-dir exports/rrf_datasets/rrf_003 --stats-only

  python scripts/build_rrf_multineg_triplets.py --batch-dir exports/rrf_datasets/rrf_003 \
      --negatives-per-anchor 3 \
      --out artifacts/twotower_rrf_triplet_bigbatch/rrf_003_multineg_k3.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from baselines.bert_frozen.text import candidate_to_text, seeker_to_text


def load_manifest_records(batch_dir: Path) -> list[dict[str, Any]]:
    manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
    return manifest["records"]


def negatives_per_query_key(batch_dir: Path) -> dict[str, int]:
    """query_key -> count of distinct negative records, for both-class keys only."""
    records = load_manifest_records(batch_dir)
    by_query: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: {"pos": [], "neg": []})
    for rec in records:
        by_query[rec["query_key"]][rec["label"]].append(rec)
    return {qk: len(g["neg"]) for qk, g in by_query.items() if g["pos"] and g["neg"]}


def print_coverage_stats(batch_dir: Path) -> None:
    counts = negatives_per_query_key(batch_dir)
    n_keys = len(counts)
    print(f"both-class query_keys: {n_keys}")
    for k in (1, 2, 3, 4, 5):
        n_full = sum(1 for c in counts.values() if c >= k)
        n_padded = n_keys - n_full
        pad_rate = n_padded / n_keys if n_keys else 0.0
        print(
            f"  k={k}: {n_full}/{n_keys} keys have >= {k} unique negatives "
            f"({pad_rate:.1%} of keys would need padding)"
        )


def build_multineg_rows(
    batch_dir: Path, *, negatives_per_anchor: int, seed: int = 42
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = load_manifest_records(batch_dir)

    by_query: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: {"pos": [], "neg": []})
    for rec in records:
        by_query[rec["query_key"]][rec["label"]].append(rec)

    pair_cache: dict[str, dict[str, Any]] = {}

    def load_pair(rec: dict[str, Any]) -> dict[str, Any]:
        path = rec["path"]
        if path not in pair_cache:
            payload = json.loads((batch_dir / path).read_text(encoding="utf-8"))
            pair_cache[path] = payload["pair"]
        return pair_cache[path]

    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    query_keys_with_both = 0
    total_padded = 0

    for query_key, groups in sorted(by_query.items()):
        pos_recs, neg_recs = groups["pos"], groups["neg"]
        if not pos_recs or not neg_recs:
            continue
        query_keys_with_both += 1

        anchor_pair = load_pair(pos_recs[0])
        anchor_text = seeker_to_text(anchor_pair["userContactFile"], anchor_pair["searchQuery"])

        unique_negs = list(neg_recs)
        rng.shuffle(unique_negs)
        n_unique = len(unique_negs)
        padded_count = max(0, negatives_per_anchor - n_unique)

        for pos_rec in pos_recs:
            pos_pair = load_pair(pos_rec)
            pos_text = candidate_to_text(pos_pair["matchContactFile"])

            if n_unique >= negatives_per_anchor:
                chosen = unique_negs[:negatives_per_anchor]
            else:
                # pad by resampling with replacement from this key's own pool
                chosen = list(unique_negs)
                while len(chosen) < negatives_per_anchor:
                    chosen.append(rng.choice(unique_negs))

            neg_pairs = [load_pair(r) for r in chosen]
            neg_ids = [p["matchContactId"] for p in neg_pairs]
            neg_texts = [candidate_to_text(p["matchContactFile"]) for p in neg_pairs]

            rows.append(
                {
                    "query_key": query_key,
                    "seeker_id": anchor_pair["userContactId"],
                    "positive_id": pos_pair["matchContactId"],
                    "negative_ids": neg_ids,
                    "anchor": anchor_text,
                    "positive": pos_text,
                    "negatives": neg_texts,
                    "n_unique_negatives": n_unique,
                    "padded_count": padded_count,
                }
            )
            total_padded += padded_count

    summary = {
        "batch_dir": str(batch_dir),
        "negatives_per_anchor": negatives_per_anchor,
        "total_query_keys": len(by_query),
        "query_keys_with_both_classes": query_keys_with_both,
        "n_rows": len(rows),
        "n_seekers": len({r["seeker_id"] for r in rows}),
        "rows_with_any_padding": sum(1 for r in rows if r["padded_count"] > 0),
        "total_padded_negative_slots": total_padded,
        "total_negative_slots": len(rows) * negatives_per_anchor,
    }
    return rows, summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-dir", type=Path, default=Path("exports/rrf_datasets/rrf_003"))
    p.add_argument("--negatives-per-anchor", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--stats-only", action="store_true", help="print coverage stats and exit, write nothing")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.stats_only:
        print_coverage_stats(args.batch_dir)
        return 0

    out = (
        args.out
        or Path("artifacts/twotower_rrf_triplet_bigbatch")
        / f"{args.batch_dir.name}_multineg_k{args.negatives_per_anchor}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    rows, summary = build_multineg_rows(args.batch_dir, negatives_per_anchor=args.negatives_per_anchor, seed=args.seed)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n")

    print(f"batch_dir: {summary['batch_dir']}")
    print(f"negatives_per_anchor: {summary['negatives_per_anchor']}")
    print(f"query_keys with both pos+neg: {summary['query_keys_with_both_classes']}")
    print(f"rows: {summary['n_rows']}")
    print(f"distinct seekers: {summary['n_seekers']}")
    padded_slot_rate = summary["total_padded_negative_slots"] / summary["total_negative_slots"] if summary["total_negative_slots"] else 0.0
    print(
        f"rows needing any padding: {summary['rows_with_any_padding']}/{summary['n_rows']}, "
        f"padded negative slots: {summary['total_padded_negative_slots']}/{summary['total_negative_slots']} ({padded_slot_rate:.1%})"
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
