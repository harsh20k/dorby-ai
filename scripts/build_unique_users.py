#!/usr/bin/env python3
"""Build data/unique_users.json by deduping pair datasets on userContactId."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SOURCE_FILES = ("dataset_positive.json", "dataset_negative.json")


def load_pairs(data_dir: Path) -> tuple[list[dict], dict[str, int]]:
    pairs: list[dict] = []
    sizes: dict[str, int] = {}
    for name in SOURCE_FILES:
        path = data_dir / name
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError(f"{path} must contain a JSON array")
        sizes[name] = len(records)
        pairs.extend(records)
    return pairs, sizes


def build_unique_users(pairs: list[dict]) -> list[dict]:
    # userContactId -> best version record + aggregates
    best: dict[str, dict] = {}
    queries: dict[str, set[str]] = {}
    pair_counts: dict[str, int] = {}

    for rec in pairs:
        uid = rec["userContactId"]
        version = int(rec.get("userContactFileVersion") or 0)
        pair_counts[uid] = pair_counts.get(uid, 0) + 1
        q = rec.get("searchQuery")
        if q:
            queries.setdefault(uid, set()).add(q)

        prev = best.get(uid)
        if prev is None or version > prev["userContactFileVersion"]:
            best[uid] = {
                "userContactId": uid,
                "userContactFileVersion": version,
                "userContactFile": rec["userContactFile"],
            }

    users: list[dict] = []
    for uid in sorted(best):
        entry = dict(best[uid])
        entry["pairCount"] = pair_counts[uid]
        qs = sorted(queries.get(uid, ()))
        if qs:
            entry["searchQueries"] = qs
        users.append(entry)
    return users


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory with dataset_*.json (default: data)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: <data-dir>/unique_users.json)",
    )
    args = parser.parse_args()
    data_dir = args.data_dir
    out_path = args.output or (data_dir / "unique_users.json")

    pairs, sizes = load_pairs(data_dir)
    users = build_unique_users(pairs)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(users, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {out_path}")
    print(f"Unique users: {len(users)}")
    for name, n in sizes.items():
        print(f"  {name}: {n}")
    print(f"  total pairs: {len(pairs)}")


if __name__ == "__main__":
    main()
