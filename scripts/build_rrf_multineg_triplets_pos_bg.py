"""Extract (anchor, positive, negative_1..negative_k) rows with BOTH sides —
seeker and candidate — trimmed to positioning + background only, no query,
no other field. Feeds `twotower_field_pos_bg/`.

Deliberate near-copy of scripts/build_rrf_multineg_triplets.py (itself
copied by three prior scripts, never imported from any of them, per the
repo's experiment-isolation rule): identical query_key grouping and
negative-sampling logic, only the two text-builder calls differ. The
free eval-time sweep (`field_pairs_sweep/`) already showed
`pos_background` is the weakest of the three field pairs on the seeker
side alone against a full-profile candidate; this script additionally
trims the candidate side to the same two fields, which no prior script in
this project has done — every other candidate-side builder in this repo
keeps the full profile.

`field_pairs_sweep.text.pos_background` is a plain profile->text function
with no seeker/candidate-specific behavior, so it is reused unmodified for
both sides here — read-only reuse of shared code, not a duplicate.

Usage:
  python scripts/build_rrf_multineg_triplets_pos_bg.py \\
      --batch-dir exports/rrf_datasets/rrf_003 --negatives-per-anchor 1 \\
      --out artifacts/twotower_field_pos_bg/rrf_003_multineg_k1_pos_bg.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from field_pairs_sweep.text import pos_background


def load_manifest_records(batch_dir: Path) -> list[dict[str, Any]]:
    manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
    return manifest["records"]


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
    n_empty_anchor = 0
    n_empty_candidate = 0

    for query_key, groups in sorted(by_query.items()):
        pos_recs, neg_recs = groups["pos"], groups["neg"]
        if not pos_recs or not neg_recs:
            continue
        query_keys_with_both += 1

        anchor_pair = load_pair(pos_recs[0])
        anchor_text = pos_background(anchor_pair["userContactFile"])
        if not anchor_text:
            n_empty_anchor += 1

        unique_negs = list(neg_recs)
        rng.shuffle(unique_negs)
        n_unique = len(unique_negs)
        padded_count = max(0, negatives_per_anchor - n_unique)

        for pos_rec in pos_recs:
            pos_pair = load_pair(pos_rec)
            pos_text = pos_background(pos_pair["matchContactFile"])
            if not pos_text:
                n_empty_candidate += 1

            if n_unique >= negatives_per_anchor:
                chosen = unique_negs[:negatives_per_anchor]
            else:
                chosen = list(unique_negs)
                while len(chosen) < negatives_per_anchor:
                    chosen.append(rng.choice(unique_negs))

            neg_pairs = [load_pair(r) for r in chosen]
            neg_ids = [p["matchContactId"] for p in neg_pairs]
            neg_texts = [pos_background(p["matchContactFile"]) for p in neg_pairs]
            for t in neg_texts:
                if not t:
                    n_empty_candidate += 1

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
        "seeker_fields": ["positioning", "background"],
        "candidate_fields": ["positioning", "background"],
        "total_query_keys": len(by_query),
        "query_keys_with_both_classes": query_keys_with_both,
        "n_rows": len(rows),
        "n_seekers": len({r["seeker_id"] for r in rows}),
        "n_empty_anchor_text": n_empty_anchor,
        "n_empty_candidate_text": n_empty_candidate,
        "rows_with_any_padding": sum(1 for r in rows if r["padded_count"] > 0),
        "total_padded_negative_slots": total_padded,
        "total_negative_slots": len(rows) * negatives_per_anchor,
    }
    return rows, summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-dir", type=Path, default=Path("exports/rrf_datasets/rrf_003"))
    p.add_argument("--negatives-per-anchor", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = (
        args.out
        or Path("artifacts/twotower_field_pos_bg")
        / f"{args.batch_dir.name}_multineg_k{args.negatives_per_anchor}_pos_bg.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    rows, summary = build_multineg_rows(args.batch_dir, negatives_per_anchor=args.negatives_per_anchor, seed=args.seed)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n")

    print(f"batch_dir: {summary['batch_dir']}")
    print(f"seeker_fields: {summary['seeker_fields']}  candidate_fields: {summary['candidate_fields']}")
    print(f"rows: {summary['n_rows']}")
    print(f"distinct seekers: {summary['n_seekers']}")
    print(f"empty anchor text: {summary['n_empty_anchor_text']}  empty candidate text: {summary['n_empty_candidate_text']}")
    padded_slot_rate = summary["total_padded_negative_slots"] / summary["total_negative_slots"] if summary["total_negative_slots"] else 0.0
    print(
        f"rows needing any padding: {summary['rows_with_any_padding']}/{summary['n_rows']}, "
        f"padded negative slots: {summary['total_padded_negative_slots']}/{summary['total_negative_slots']} ({padded_slot_rate:.1%})"
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
