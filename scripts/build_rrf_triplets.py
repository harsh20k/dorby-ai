"""Extract (anchor, positive, negative) triplets from a pairing_rrf batch.

Groups a batch's judged pairs by `query_key` (one seeker + one of its
`lookingFor`-derived queries — the unit that determines the anchor text, since
`seeker_text = seeker_to_text(userContactFile, searchQuery)` depends on both).
For every query_key that has at least one `pos` and one `neg` judged pair, emits
one triplet per (pos, neg) combination.

These labels are an LLM judge's opinion on synthetic profiles, not real
accept/decline outcomes (see manifest.json's `promotion_note`) — this output is
training data for an experimental arm, not something promoted into
data/dataset_*.json.

Usage:
  python scripts/build_rrf_triplets.py --batch-dir exports/rrf_datasets/rrf_003 \
      --out artifacts/twotower/rrf_003_triplets.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from baselines.bert_frozen.text import candidate_to_text, seeker_to_text


def load_manifest_records(batch_dir: Path) -> list[dict[str, Any]]:
    manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
    return manifest["records"]


def build_triplets(batch_dir: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    records = load_manifest_records(batch_dir)

    by_query: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {"pos": [], "neg": []}
    )
    for rec in records:
        by_query[rec["query_key"]][rec["label"]].append(rec)

    pair_cache: dict[str, dict[str, Any]] = {}

    def load_pair(rec: dict[str, Any]) -> dict[str, Any]:
        path = rec["path"]
        if path not in pair_cache:
            payload = json.loads((batch_dir / path).read_text(encoding="utf-8"))
            pair_cache[path] = payload["pair"]
        return pair_cache[path]

    triplets: list[dict[str, str]] = []
    query_keys_with_both = 0
    for query_key, groups in by_query.items():
        pos_recs, neg_recs = groups["pos"], groups["neg"]
        if not pos_recs or not neg_recs:
            continue
        query_keys_with_both += 1

        anchor_pair = load_pair(pos_recs[0])
        anchor_text = seeker_to_text(anchor_pair["userContactFile"], anchor_pair["searchQuery"])

        for pos_rec in pos_recs:
            pos_pair = load_pair(pos_rec)
            pos_text = candidate_to_text(pos_pair["matchContactFile"])
            for neg_rec in neg_recs:
                neg_pair = load_pair(neg_rec)
                neg_text = candidate_to_text(neg_pair["matchContactFile"])
                triplets.append(
                    {
                        "query_key": query_key,
                        "seeker_id": anchor_pair["userContactId"],
                        "positive_id": pos_pair["matchContactId"],
                        "negative_id": neg_pair["matchContactId"],
                        "anchor": anchor_text,
                        "positive": pos_text,
                        "negative": neg_text,
                    }
                )

    summary = {
        "batch_dir": str(batch_dir),
        "total_query_keys": len(by_query),
        "query_keys_with_both_classes": query_keys_with_both,
        "n_triplets": len(triplets),
        "n_seekers": len({t["seeker_id"] for t in triplets}),
    }
    return triplets, summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-dir", type=Path, default=Path("exports/rrf_datasets/rrf_003"))
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = args.out or Path("artifacts/twotower_rrf_triplet") / f"{args.batch_dir.name}_triplets.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    triplets, summary = build_triplets(args.batch_dir)
    out.write_text(json.dumps({"summary": summary, "triplets": triplets}, indent=2) + "\n")

    print(f"batch_dir: {summary['batch_dir']}")
    print(f"query_keys: {summary['total_query_keys']}")
    print(f"query_keys with both pos+neg: {summary['query_keys_with_both_classes']}")
    print(f"triplets: {summary['n_triplets']}")
    print(f"distinct seekers: {summary['n_seekers']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
