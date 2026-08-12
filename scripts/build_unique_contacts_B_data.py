#!/usr/bin/env python3
"""Dedupe locked B-data into unique contacts via positioning hash + fallback.

Identity:
  nonempty positioning → sha256("positioning\\0" + stripped text)
  else first nonempty of background, lookingFor, locationAvailability,
  introPreferences, personalPreferences, meetingAndSchedulingPreferences
  all-empty profiles are dropped.

Does not modify data/B-data.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "B-data.json"
DEFAULT_OUT = ROOT / "data" / "unique_contacts_B_data.json"
DEFAULT_STATS = ROOT / "artifacts" / "bdata_unique_contacts" / "stats.json"

PROFILE_FIELDS = (
    "positioning",
    "background",
    "lookingFor",
    "locationAvailability",
    "introPreferences",
    "personalPreferences",
    "meetingAndSchedulingPreferences",
)


def _strip(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _richness(contact_file: dict[str, Any] | None) -> int:
    if not isinstance(contact_file, dict):
        return 0
    return sum(len(_strip(v)) for v in contact_file.values())


def identity_key(contact_file: dict[str, Any] | None) -> tuple[str, str] | None:
    """Return (field, hex digest) or None if the profile is entirely empty."""
    cf = contact_file if isinstance(contact_file, dict) else {}
    for field in PROFILE_FIELDS:
        text = _strip(cf.get(field))
        if text:
            digest = hashlib.sha256(f"{field}\0{text}".encode()).hexdigest()
            return field, digest
    return None


def _ensure_bucket(store: dict[str, dict[str, Any]], field: str, digest: str) -> dict[str, Any]:
    bucket = store.get(digest)
    if bucket is None:
        bucket = {
            "identityKey": digest,
            "identityField": field,
            "contactIds": set(),
            "queries": set(),
            "seekerCount": 0,
            "matchCount": 0,
            "matchStatuses": Counter(),
            "matchTypes": Counter(),
            "contactFile": {},
            "_richness": -1,
        }
        store[digest] = bucket
    return bucket


def _ingest(
    store: dict[str, dict[str, Any]],
    contact_file: Any,
    *,
    role: str,
    contact_id: str | None,
    query: str | None,
    status: str | None = None,
    match_type: str | None = None,
) -> str | None:
    ident = identity_key(contact_file if isinstance(contact_file, dict) else None)
    if ident is None:
        return "dropped_empty"
    field, digest = ident
    bucket = _ensure_bucket(store, field, digest)
    cf = contact_file if isinstance(contact_file, dict) else {}
    score = _richness(cf)
    if score > bucket["_richness"]:
        bucket["contactFile"] = cf
        bucket["_richness"] = score
        bucket["identityField"] = field
    if contact_id:
        bucket["contactIds"].add(contact_id)
    if query and query.strip():
        bucket["queries"].add(query.strip())
    if role == "seeker":
        bucket["seekerCount"] += 1
    else:
        bucket["matchCount"] += 1
        if status:
            bucket["matchStatuses"][status] += 1
        if match_type:
            bucket["matchTypes"][match_type] += 1
    return None


def build_unique_contacts(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    store: dict[str, dict[str, Any]] = {}
    dropped_empty = 0
    n_seeker_slots = 0
    n_match_slots = 0

    for rec in rows:
        n_seeker_slots += 1
        if _ingest(
            store,
            rec.get("contactFile"),
            role="seeker",
            contact_id=str(rec.get("contactId") or "") or None,
            query=rec.get("query") if isinstance(rec.get("query"), str) else None,
        ) == "dropped_empty":
            dropped_empty += 1
        for match in rec.get("matches") or []:
            if not isinstance(match, dict):
                continue
            n_match_slots += 1
            if _ingest(
                store,
                match.get("contactFile"),
                role="match",
                contact_id=None,
                query=None,
                status=str(match.get("status") or "") or None,
                match_type=str(match.get("matchType") or "") or None,
            ) == "dropped_empty":
                dropped_empty += 1

    contacts: list[dict[str, Any]] = []
    identity_field_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    with_contact_id = 0
    field_present: Counter[str] = Counter()

    for digest in sorted(store):
        b = store[digest]
        ids = sorted(b["contactIds"])
        qs = sorted(b["queries"])
        seeker_n = int(b["seekerCount"])
        match_n = int(b["matchCount"])
        if seeker_n and match_n:
            role = "both"
        elif seeker_n:
            role = "seeker"
        else:
            role = "candidate"
        role_counts[role] += 1
        identity_field_counts[str(b["identityField"])] += 1
        if ids:
            with_contact_id += 1
        cf = b["contactFile"] if isinstance(b["contactFile"], dict) else {}
        for field in PROFILE_FIELDS:
            if _strip(cf.get(field)):
                field_present[field] += 1
        entry: dict[str, Any] = {
            "identityKey": digest,
            "identityField": b["identityField"],
            "role": role,
            "seekerCount": seeker_n,
            "matchCount": match_n,
            "contactFile": cf,
        }
        if ids:
            entry["contactIds"] = ids
        if qs:
            entry["searchQueries"] = qs
        statuses = dict(b["matchStatuses"])
        if statuses:
            entry["matchStatuses"] = statuses
        types = dict(b["matchTypes"])
        if types:
            entry["matchTypes"] = types
        contacts.append(entry)

    n = len(contacts)
    stats = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "source": "data/B-data.json",
        "identity_rule": (
            "sha256(field + NUL + stripped text); field = positioning if nonempty, "
            "else first nonempty of background/lookingFor/locationAvailability/"
            "introPreferences/personalPreferences/meetingAndSchedulingPreferences"
        ),
        "n_source_rows": len(rows),
        "n_seeker_slots": n_seeker_slots,
        "n_match_slots": n_match_slots,
        "n_dropped_all_empty": dropped_empty,
        "n_unique_contacts": n,
        "n_with_contact_id": with_contact_id,
        "n_without_contact_id": n - with_contact_id,
        "role_counts": dict(role_counts),
        "identity_field_counts": dict(identity_field_counts),
        "field_coverage": {
            f: {"n": field_present[f], "pct": round(100.0 * field_present[f] / n, 2) if n else 0.0}
            for f in PROFILE_FIELDS
        },
    }
    return contacts, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    args = parser.parse_args()

    source = args.source
    if not source.is_file():
        raise SystemExit(f"Missing source: {source}")
    if source.stat().st_mode & 0o222:
        raise SystemExit(
            f"Refusing to proceed: {source} is writable. Lock it first (chmod 444)."
        )

    rows = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise SystemExit(f"{source} must be a JSON array")

    contacts, stats = build_unique_contacts(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(contacts, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"Read {source} ({len(rows)} rows, locked/read-only)")
    print(f"Wrote {args.output} ({len(contacts)} unique contacts)")
    print(f"Wrote {args.stats}")
    print(
        f"roles={stats['role_counts']} "
        f"identity_fields={stats['identity_field_counts']} "
        f"dropped_empty={stats['n_dropped_all_empty']}"
    )


if __name__ == "__main__":
    main()
