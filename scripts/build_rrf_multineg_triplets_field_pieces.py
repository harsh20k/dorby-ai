"""Extract rows with the seeker side kept as SEPARATE pieces — search query,
`lookingFor`, `positioning` — instead of one flat concatenated string. Feeds
the `twotower_field_gate/` experiment: one shared tower embeds each piece
separately, then a small learned gate combines them into one seeker vector.

Deliberate near-copy of scripts/build_rrf_multineg_triplets_query_only.py
(itself a copy of the original), not imported from either, per the repo's
experiment-isolation rule. The candidate side (`positive`/`negatives`) is
unchanged `candidate_to_text` — the novelty here is scoped to the seeker
side, per the design question ("can the first tower ... learn to combine
them").

    python scripts/build_rrf_multineg_triplets_field_pieces.py \\
        --batch-dir exports/rrf_datasets/rrf_003 --negatives-per-anchor 1 \\
        --out artifacts/twotower_field_gate/rrf_003_multineg_k1_field_pieces.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from baselines.bert_frozen.text import candidate_to_text
from query_weighted.text import query_only

PIECE_FIELDS = ("lookingFor", "positioning")


def load_manifest_records(batch_dir: Path) -> list[dict[str, Any]]:
    manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
    return manifest["records"]


def _piece_text(field: str, value: str) -> str:
    return f"{field}: {value.strip()}"


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
    n_missing_piece = 0

    for query_key, groups in sorted(by_query.items()):
        pos_recs, neg_recs = groups["pos"], groups["neg"]
        if not pos_recs or not neg_recs:
            continue
        query_keys_with_both += 1

        anchor_pair = load_pair(pos_recs[0])
        profile = anchor_pair["userContactFile"]

        query_text = query_only(profile, anchor_pair["searchQuery"])
        pieces = {"query": query_text}
        for field in PIECE_FIELDS:
            value = (profile.get(field) or "").strip()
            if not value:
                n_missing_piece += 1
                value = query_text  # rare fallback; verified 0/400 in this batch
            pieces[field] = _piece_text(field, value)

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
                    "pieces": pieces,  # {"query": ..., "lookingFor": ..., "positioning": ...}
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
        "piece_fields": ["query", *PIECE_FIELDS],
        "total_query_keys": len(by_query),
        "query_keys_with_both_classes": query_keys_with_both,
        "n_rows": len(rows),
        "n_seekers": len({r["seeker_id"] for r in rows}),
        "n_missing_piece_fallback": n_missing_piece,
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
        or Path("artifacts/twotower_field_gate")
        / f"{args.batch_dir.name}_multineg_k{args.negatives_per_anchor}_field_pieces.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    rows, summary = build_multineg_rows(args.batch_dir, negatives_per_anchor=args.negatives_per_anchor, seed=args.seed)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n")

    print(f"batch_dir: {summary['batch_dir']}")
    print(f"piece_fields: {summary['piece_fields']}")
    print(f"rows: {summary['n_rows']}")
    print(f"distinct seekers: {summary['n_seekers']}")
    print(f"missing-piece fallback count: {summary['n_missing_piece_fallback']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
