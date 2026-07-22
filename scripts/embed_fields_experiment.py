"""Embed each profile field separately (voyage-4-nano, local) for a small sample
of non-holdout contacts, and cache the results for reuse.

First step of the "field-pair matching" idea: instead of one embedding per
contact (the field-tagged blob every existing baseline uses), embed each
field on its own so later experiments can compare specific field pairs
(e.g. seeker lookingFor vs. candidate positioning/background) instead of
one blob-vs-blob similarity. This script only does the embedding + caching;
it doesn't score anything yet.

Usage:
    python scripts/embed_fields_experiment.py --data-dir data --n-contacts 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from baselines.voyage_nano.encode import VoyageNanoEncoder

FIELDS = [
    "positioning",
    "background",
    "lookingFor",
    "notes",
    "locationAvailability",
    "introPreferences",
    "personalPreferences",
    "meetingAndSchedulingPreferences",
]


def load_split(data_dir: Path) -> dict[str, Any]:
    return json.loads((data_dir / "synthetic" / "seed_split.json").read_text())


def build_contact_profiles(
    positives: list[dict[str, Any]], negatives: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """First-seen profile dict per contact id, from either seeker or candidate side."""
    profiles: dict[str, dict[str, Any]] = {}
    for record in positives + negatives:
        uid, mid = record["userContactId"], record["matchContactId"]
        profiles.setdefault(uid, record["userContactFile"])
        profiles.setdefault(mid, record["matchContactFile"])
    return profiles


def pick_sample_contacts(
    train_user_ids: list[str], profiles: dict[str, dict[str, Any]], n: int
) -> list[str]:
    sample = [uid for uid in train_user_ids if uid in profiles][:n]
    if len(sample) < n:
        raise ValueError(f"only found {len(sample)} train contacts with a profile, need {n}")
    return sample


def run(data_dir: Path, n_contacts: int, cache_dir: Path) -> dict[str, Any]:
    split = load_split(data_dir)
    positives = json.loads((data_dir / "dataset_positive.json").read_text())
    negatives = json.loads((data_dir / "dataset_negative.json").read_text())

    profiles = build_contact_profiles(positives, negatives)
    contact_ids = pick_sample_contacts(split["train_user_ids"], profiles, n_contacts)
    print(f"sampled {len(contact_ids)} non-holdout contacts: {contact_ids}")

    encoder = VoyageNanoEncoder(cache_dir=cache_dir)

    index: dict[str, dict[str, Any]] = {}
    for contact_id in contact_ids:
        profile = profiles[contact_id]
        index[contact_id] = {}
        for field in FIELDS:
            text = profile.get(field)
            if not text or not str(text).strip():
                index[contact_id][field] = {"present": False}
                continue

            cache_name = f"{contact_id}__{field}"
            emb = encoder.encode(
                [str(text)],
                role="document",
                batch_size=1,
                show_progress=False,
                cache_name=cache_name,
            )
            index[contact_id][field] = {
                "present": True,
                "cache_name": cache_name,
                "shape": list(emb.shape),
                "chars": len(str(text)),
            }
            print(f"  {contact_id} / {field:32s} -> {emb.shape} ({len(str(text))} chars)")

    index_path = cache_dir / "field_index.json"
    index_path.write_text(json.dumps(index, indent=2))
    print(f"\nwrote {index_path}")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--n-contacts", type=int, default=10)
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/field_embeddings/nano"))
    args = parser.parse_args()
    run(args.data_dir, args.n_contacts, args.cache_dir)


if __name__ == "__main__":
    main()
