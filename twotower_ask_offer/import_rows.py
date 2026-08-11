"""Freeze the ask/offer training rows from voyage_gemini_ctrl's source data.

Isolation rule: input data is copied, never read live. This script is the one
place that touches `artifacts/twotower_voyage_gemini_ctrl/` and
`artifacts/pairing_voyage_gemini/smoke_test_002/staged/` (both read-only,
neither ever written to). It writes one frozen file with provenance
(source paths + content hashes) that `twotower_ask_offer.data` loads instead.

Same 3008 rows, same query_keys, same (seeker, positive, negative) ids as
`voyage_gemini_ctrl_001` trained on — kept identical on purpose so this
experiment differs from that one in exactly one place (two towers + the
reciprocal loss), not in the training population too. The only thing this
script does differently is resolve each id back to its raw profile dict
(positioning/background/lookingFor as separate fields) instead of the
pre-baked full-profile text `voyage_gemini_ctrl` trained on, since the ask/offer
split needs the fields separately, not concatenated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SOURCE_ROWS_PATH = Path("artifacts/twotower_voyage_gemini_ctrl/voyage_gemini_smoke002_multineg_k1.json")
SOURCE_STAGED_DIR = Path("artifacts/pairing_voyage_gemini/smoke_test_002/staged")
FROZEN_PATH = Path("artifacts/twotower_ask_offer/ask_offer_rows.json")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _load_staged_pair(staged_dir: Path, query_key: str, candidate_id: str) -> dict[str, Any]:
    path = staged_dir / f"{query_key}__{candidate_id}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["pair"]


def build_frozen_rows(
    *,
    source_rows_path: Path = SOURCE_ROWS_PATH,
    staged_dir: Path = SOURCE_STAGED_DIR,
) -> dict[str, Any]:
    source = json.loads(source_rows_path.read_text(encoding="utf-8"))
    src_rows = source["rows"]

    frozen_rows: list[dict[str, Any]] = []
    profile_cache: dict[tuple[str, str], dict[str, Any]] = {}

    def cached_pair(query_key: str, candidate_id: str) -> dict[str, Any]:
        key = (query_key, candidate_id)
        if key not in profile_cache:
            profile_cache[key] = _load_staged_pair(staged_dir, query_key, candidate_id)
        return profile_cache[key]

    for r in src_rows:
        query_key = r["query_key"]
        seeker_id = r["seeker_id"]
        positive_id = r["positive_id"]
        negative_ids = r["negative_ids"]

        pos_pair = cached_pair(query_key, positive_id)
        assert pos_pair["userContactId"] == seeker_id, (query_key, positive_id)
        assert pos_pair["matchContactId"] == positive_id, (query_key, positive_id)

        negative_profiles = []
        for neg_id in negative_ids:
            neg_pair = cached_pair(query_key, neg_id)
            assert neg_pair["matchContactId"] == neg_id, (query_key, neg_id)
            negative_profiles.append(neg_pair["matchContactFile"])

        frozen_rows.append(
            {
                "query_key": query_key,
                "seeker_id": seeker_id,
                "positive_id": positive_id,
                "negative_ids": negative_ids,
                "search_query": pos_pair["searchQuery"],
                "seeker_profile": pos_pair["userContactFile"],
                "positive_profile": pos_pair["matchContactFile"],
                "negative_profiles": negative_profiles,
            }
        )

    return {
        "provenance": {
            "source_rows_path": str(source_rows_path),
            "source_rows_sha256": _sha256_file(source_rows_path),
            "source_staged_dir": str(staged_dir),
            "n_staged_pairs_resolved": len(profile_cache),
            "source_batch_summary": source["summary"],
        },
        "n_rows": len(frozen_rows),
        "rows": frozen_rows,
    }


def verify(frozen_path: Path = FROZEN_PATH) -> bool:
    """Re-derive from source and confirm the frozen copy still matches."""
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    prov = frozen["provenance"]
    source_path = Path(prov["source_rows_path"])
    current_hash = _sha256_file(source_path)
    if current_hash != prov["source_rows_sha256"]:
        print(f"MISMATCH: source {source_path} hash changed since import "
              f"({prov['source_rows_sha256']} -> {current_hash})")
        return False
    rebuilt = build_frozen_rows(source_rows_path=source_path, staged_dir=Path(prov["source_staged_dir"]))
    if rebuilt["rows"] != frozen["rows"]:
        print("MISMATCH: rebuilt rows differ from frozen copy")
        return False
    print(f"OK: {frozen_path} matches source ({frozen['n_rows']} rows)")
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verify", action="store_true", help="verify frozen copy against source instead of writing")
    p.add_argument("--out", type=Path, default=FROZEN_PATH)
    args = p.parse_args()

    if args.verify:
        ok = verify(args.out)
        return 0 if ok else 1

    frozen = build_frozen_rows()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: {frozen['n_rows']} rows, "
          f"{frozen['provenance']['n_staged_pairs_resolved']} staged pairs resolved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
