"""Freeze the identity of the 200 real pairs, and verify the source hasn't moved.

The isolation rule asks each experiment to copy its input data into its own
namespace with provenance and a ``--verify`` mode. Here the input is real
contact profiles under the gitignored ``data/``, so copying the *content* into a
tracked package directory would commit real user data to git.

The compromise: freeze pair identity + per-pair SHA-256 digests. That still
detects any drift in the pairs this experiment ran against, without storing a
single profile field.

    python -m eval_real_full.freeze --data-dir data          # write manifest
    python -m eval_real_full.freeze --data-dir data --verify  # check for drift
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from twotower.data import load_canonical_pairs
from synth_pipeline.split import load_split

MANIFEST_PATH = Path(__file__).parent / "data_frozen" / "real_200_manifest.json"


def is_real(pair: dict[str, Any]) -> bool:
    """A pair is real iff neither contact id carries the synthetic prefix."""
    return not (
        str(pair["userContactId"]).startswith("cmsynth")
        or str(pair["matchContactId"]).startswith("cmsynth")
    )


def pair_id(pair: dict[str, Any], label: str) -> str:
    return f"{label}:{pair['userContactId']}:{pair['matchContactId']}"


def pair_digest(pair: dict[str, Any]) -> str:
    blob = json.dumps(pair, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def pair_key(pair_id_str: str) -> str:
    """Opaque, stable key for a pair — never the raw contact ids.

    ``data/`` is gitignored precisely so real Boardy contact identifiers stay
    out of the repository, and no tracked file contains one. This manifest is
    tracked, so it stores a hash of the pair id rather than the id itself. Drift
    detection is unaffected (the key is deterministic), and a changed pair can
    still be pinpointed — just by opaque key rather than by name.
    """
    return hashlib.sha256(pair_id_str.encode()).hexdigest()[:16]


def collect(data_dir: Path, split_path: Path) -> dict[str, Any]:
    positives, negatives = load_canonical_pairs(data_dir)
    split = load_split(split_path)
    eval_pair_ids = set(split["eval_pair_ids"])
    train_pair_ids = set(split["train_pair_ids"])

    entries: list[dict[str, str]] = []
    for label, rows in (("pos", positives), ("neg", negatives)):
        for pair in rows:
            if not is_real(pair):
                continue
            pid = pair_id(pair, label)
            if pid in eval_pair_ids:
                subset = "holdout"
            elif pid in train_pair_ids:
                subset = "train"
            else:
                # A real pair in neither frozen list would silently change what
                # "all 200" means; refuse rather than guess.
                raise KeyError(f"real pair in neither frozen split list: {pid}")
            entries.append(
                {
                    "pair_key": pair_key(pid),
                    "label": label,
                    "subset": subset,
                    "sha256": pair_digest(pair),
                }
            )

    entries.sort(key=lambda e: e["pair_key"])
    combined = hashlib.sha256(
        json.dumps([e["sha256"] for e in entries], separators=(",", ":")).encode()
    ).hexdigest()[:16]

    counts: dict[str, int] = {}
    for e in entries:
        counts[e["subset"]] = counts.get(e["subset"], 0) + 1
        counts[e["label"]] = counts.get(e["label"], 0) + 1
    counts["total"] = len(entries)

    return {
        "source": {
            "data_dir": str(data_dir),
            "positives_file": "dataset_positive.json",
            "negatives_file": "dataset_negative.json",
            "split_path": str(split_path),
            "split_hash": split.get("split_hash"),
        },
        "note": (
            "Digests only — no profile content and no real contact ids are stored "
            "here. data/ is gitignored real user data and no tracked file in this "
            "repo contains a contact id; this manifest keeps that true by hashing "
            "pair identity into pair_key. See eval_real_full/__init__.py."
        ),
        "counts": counts,
        "combined_hash": combined,
        "pairs": entries,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--split-path", type=Path, default=None)
    p.add_argument("--verify", action="store_true", help="compare against the frozen manifest")
    args = p.parse_args(argv)
    split_path = args.split_path or args.data_dir / "synthetic" / "seed_split.json"

    current = collect(args.data_dir, split_path)
    print(f"real pairs found: {current['counts']}")
    print(f"combined hash:    {current['combined_hash']}")

    if not args.verify:
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST_PATH.write_text(json.dumps(current, indent=2) + "\n")
        print(f"wrote {MANIFEST_PATH}")
        return 0

    if not MANIFEST_PATH.exists():
        print(f"FAIL: no frozen manifest at {MANIFEST_PATH}; run without --verify first")
        return 1
    frozen = json.loads(MANIFEST_PATH.read_text())
    if frozen["combined_hash"] == current["combined_hash"]:
        print("OK: source data matches the frozen manifest exactly.")
        return 0

    print("FAIL: source data has drifted since import.")
    fz = {e["pair_key"]: e["sha256"] for e in frozen["pairs"]}
    cu = {e["pair_key"]: e["sha256"] for e in current["pairs"]}
    for key in sorted(set(fz) - set(cu)):
        print(f"  removed: pair_key={key}")
    for key in sorted(set(cu) - set(fz)):
        print(f"  added:   pair_key={key}")
    for key in sorted(set(fz) & set(cu)):
        if fz[key] != cu[key]:
            print(f"  changed: pair_key={key}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
