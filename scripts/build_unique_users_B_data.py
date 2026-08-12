#!/usr/bin/env python3
"""Build unique_users_B_data.json from locked data/B-data.json (read-only).

B-data rows look like:
  {contactId, query, contactFile, matches:[...]}

This mirrors scripts/build_unique_users.py's seeker-side unique_users shape
without touching B-data.json. Match contacts have no id and are not included.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "B-data.json"
DEFAULT_OUT = ROOT / "data" / "unique_users_B_data.json"


def _profile_richness(contact_file: dict[str, Any] | None) -> int:
    if not isinstance(contact_file, dict):
        return 0
    total = 0
    for value in contact_file.values():
        if isinstance(value, str):
            total += len(value)
        elif value is not None:
            total += len(json.dumps(value, ensure_ascii=False))
    return total


def build_unique_users(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    queries: dict[str, set[str]] = {}
    pair_counts: dict[str, int] = {}
    richness: dict[str, int] = {}

    for rec in rows:
        uid = rec.get("contactId")
        if not uid:
            raise ValueError("B-data row missing contactId")
        pair_counts[uid] = pair_counts.get(uid, 0) + 1
        q = rec.get("query")
        if isinstance(q, str) and q.strip():
            queries.setdefault(uid, set()).add(q.strip())

        contact_file = rec.get("contactFile")
        score = _profile_richness(contact_file if isinstance(contact_file, dict) else None)
        prev = best.get(uid)
        if prev is None or score > richness.get(uid, -1):
            best[uid] = {
                "userContactId": uid,
                "userContactFileVersion": 0,
                "userContactFile": contact_file if isinstance(contact_file, dict) else {},
            }
            richness[uid] = score

    users: list[dict[str, Any]] = []
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
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    source = args.source
    if not source.is_file():
        raise SystemExit(f"Missing source: {source}")
    if source.stat().st_mode & 0o222:
        raise SystemExit(
            f"Refusing to proceed: {source} is writable. "
            "Lock it first (chmod 444) — B-data is immutable."
        )

    rows = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise SystemExit(f"{source} must be a JSON array")

    users = build_unique_users(rows)
    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(users, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Read {source} ({len(rows)} rows, locked/read-only)")
    print(f"Wrote {out_path}")
    print(f"Unique users: {len(users)}")


if __name__ == "__main__":
    main()
