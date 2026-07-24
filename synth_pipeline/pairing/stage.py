"""Write a pairing batch to its own isolated namespace.

Batches live under `artifacts/pairing/<batch_id>/` and are NEVER promoted into
`data/dataset_*.json`. That is the point: labels here come from a scorer, not a
judge or a human, so mixing them into the canonical dataset would repeat the
batch_500_001 mistake in a new form. Keeping each experiment self-contained is
also what makes them cleanly comparable.

Envelopes keep the exact key set that `synth_pipeline/nodes/writer.py` writes, so
the existing review browser and `promote.py` remain available later without rework
— they are simply not in this pipeline's path.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from synth_pipeline.schema import jaccard, tokenize, validate_pair_schema

# Must match synth_pipeline/nodes/writer.py's envelope exactly — tests assert this.
ENVELOPE_KEYS = {
    "pair",
    "label",
    "failure_mode",
    "seed_pair_id",
    "seed_user_id",
    "batch_id",
    "split_hash",
    "metadata",
    "qc",
    "human_review",
}


def batch_dir(artifacts_dir: Path, batch_id: str) -> Path:
    return Path(artifacts_dir) / "pairing" / batch_id


def build_pair(candidate) -> dict[str, Any]:
    """The 7 PAIR_KEYS. No `label` field — membership is the label, as in real data."""
    return {
        "userContactId": candidate.seeker.contact_id,
        "matchContactId": candidate.candidate.contact_id,
        "userContactFileVersion": 1,
        "matchContactFileVersion": 1,
        "searchQuery": candidate.query,
        "userContactFile": candidate.seeker.as_contact_file(),
        "matchContactFile": candidate.candidate.as_contact_file(),
    }


def build_envelope(
    candidate,
    *,
    batch_id: str,
    label: str | None,
    score: float,
    split_hash: str,
    scorer_meta: dict[str, Any],
    drop_reason: str | None = None,
) -> dict[str, Any]:
    pair = build_pair(candidate)
    errors = validate_pair_schema(pair)
    if errors:
        raise ValueError(f"built an invalid pair: {errors[:3]}")

    query_tokens = tokenize(candidate.query)
    cand_blob = " ".join(
        v for v in candidate.candidate.profile.values() if isinstance(v, str)
    )

    return {
        "pair": pair,
        "label": label,
        # Negatives here have no failure_mode: nothing diagnosed *why* the match
        # is bad. A scorer produces a number, not a reason. Left null rather than
        # guessed, so downstream per-mode analysis can't be misled.
        "failure_mode": None,
        "seed_pair_id": None,
        "seed_user_id": None,
        "batch_id": batch_id,
        "split_hash": split_hash,
        "metadata": {
            "source": "pairing",
            "profile_run": candidate.seeker.source_run,
            "seeker_profile_id": candidate.seeker.profile_id,
            "candidate_profile_id": candidate.candidate.profile_id,
            "seeker_archetype": candidate.seeker.archetype,
            "candidate_archetype": candidate.candidate.archetype,
            "same_archetype": candidate.same_archetype,
            "query_index": candidate.query_index,
            "tfidf_query_cosine": candidate.cosine,
            "select_rank": candidate.rank,
            "band": candidate.band,
            "labeler": "hybrid_tfidf_voyage",
            "scorer": scorer_meta,
        },
        "qc": {
            "fusion_score": float(score),
            "query_match_jaccard": jaccard(query_tokens, tokenize(cand_blob)),
            "drop_reason": drop_reason,
        },
        "human_review": None,
    }


def write_batch(
    artifacts_dir: Path,
    batch_id: str,
    *,
    envelopes: list[dict[str, Any]],
    profiles: list,
    queries: dict[str, list[str]],
    config: dict[str, Any],
    thresholds: dict[str, Any],
    scorer_meta: dict[str, Any],
    usage: dict[str, int],
) -> dict[str, Any]:
    root = batch_dir(artifacts_dir, batch_id)
    (root / "pairs").mkdir(parents=True, exist_ok=True)
    (root / "excluded").mkdir(parents=True, exist_ok=True)

    counts = {"pos": 0, "neg": 0, "excluded": 0}
    for env in envelopes:
        label = env["label"]
        pair = env["pair"]
        # {label}_{seeker}_{candidate}: the old {label}_{matchContactId} scheme
        # collides here, because one candidate profile appears in many pairs.
        stem = f"{label or 'excluded'}_{pair['userContactId']}_{pair['matchContactId']}"
        sub = "pairs" if label else "excluded"
        (root / sub / f"{stem}.json").write_text(
            json.dumps(env, indent=2) + "\n", encoding="utf-8"
        )
        counts[label if label else "excluded"] += 1

    (root / "profiles.json").write_text(
        json.dumps(
            {
                p.contact_id: {
                    "profile_id": p.profile_id,
                    "archetype": p.archetype,
                    "source_run": p.source_run,
                    "profile": p.profile,
                }
                for p in profiles
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "queries.json").write_text(
        json.dumps(queries, indent=2) + "\n", encoding="utf-8"
    )

    manifest = {
        "batch_id": batch_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "counts": {**counts, "profiles": len(profiles), "total_pairs": len(envelopes)},
        "thresholds": thresholds,
        "scorer": scorer_meta,
        "usage": usage,
        "note": (
            "Labels come from the hybrid TF-IDF+Voyage-nano scorer, not a judge or a "
            "human. Provisional — not training data. Never promoted into "
            "data/dataset_*.json."
        ),
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
