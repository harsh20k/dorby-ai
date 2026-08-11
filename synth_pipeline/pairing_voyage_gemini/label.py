"""Turn judge verdicts into staged pair envelopes.

There is no deadband. Every pair that reaches the judge becomes a labeled pair:
``yes`` is a positive, ``no`` is a hard negative. The trade is explicit — the
judge decides at 0.5942 accuracy on the real hard slice, so borderline pairs will
carry some wrong labels — and it is taken because it makes calls-per-labeled-pair
exactly 1.0, with no spend on a pair that does not become data.

The negatives are the point. A candidate that dense retrieval ranked in its top
ten and that the judge still turned down is close to this project's target
population: something a production matcher recommended and a human declined.
Constructed mismatches are not in the same class.

Envelope shape matches ``synth_pipeline/pairing/stage.py`` exactly, including the
deliberate ``failure_mode: None`` — a yes/no verdict says a pair is bad, not
*which axis* is bad, and guessing a mode would mislead per-mode analysis. What
this pipeline adds over the scorer-based one is the judge's own reasoning text,
kept for review.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from synth_pipeline.pairing.stage import ENVELOPE_KEYS
from synth_pipeline.schema import validate_pair_schema

LABEL_POSITIVE = "pos"
LABEL_NEGATIVE = "neg"


@dataclass
class JudgedPair:
    """One shortlisted pair with its verdict attached."""

    seeker_id: str
    candidate_id: str
    query_key: str
    query_text: str
    section_index: int
    seeker_profile: Mapping[str, Any]
    candidate_profile: Mapping[str, Any]
    verdict: Mapping[str, Any]
    dense: Mapping[str, Any]

    @property
    def pair_key(self) -> str:
        return f"{self.query_key}__{self.candidate_id}"

    @property
    def label(self) -> str | None:
        match = str(self.verdict.get("match") or "").lower()
        if match == "yes":
            return LABEL_POSITIVE
        if match == "no":
            return LABEL_NEGATIVE
        return None


def build_pair(judged: JudgedPair) -> dict[str, Any]:
    """The 7 pair keys. No ``label`` field — membership is the label, as in real data."""
    return {
        "userContactId": judged.seeker_id,
        "matchContactId": judged.candidate_id,
        "userContactFileVersion": 1,
        "matchContactFileVersion": 1,
        "searchQuery": judged.query_text,
        "userContactFile": dict(judged.seeker_profile),
        "matchContactFile": dict(judged.candidate_profile),
    }


def build_envelope(
    judged: JudgedPair,
    *,
    batch_id: str,
    split_hash: str,
    labeler_meta: Mapping[str, Any],
    drop_reason: str | None = None,
) -> dict[str, Any]:
    pair = build_pair(judged)
    errors = validate_pair_schema(pair)
    if errors:
        raise ValueError(f"built an invalid pair: {errors[:3]}")

    envelope = {
        "pair": pair,
        "label": judged.label,
        "failure_mode": None,
        "seed_pair_id": None,
        "seed_user_id": None,
        "batch_id": batch_id,
        "split_hash": split_hash,
        "metadata": {
            "source": "pairing_voyage_gemini",
            "query_key": judged.query_key,
            "section_index": judged.section_index,
            "labeler": dict(labeler_meta),
            "dense": dict(judged.dense),
            "drop_reason": drop_reason,
        },
        "qc": {
            "judge_match": judged.verdict.get("match"),
            # Recorded for audit only. The judge experiment measured stated
            # confidence at 88.6 when right and 88.2 when wrong — no
            # discrimination — so nothing downstream may gate on this.
            "judge_confidence_unused": judged.verdict.get("confidence"),
            "judge_reasoning": judged.verdict.get("reasoning"),
        },
        "human_review": {"verdict": None, "notes": None},
    }
    extra = set(envelope) - ENVELOPE_KEYS
    if extra:
        raise ValueError(f"envelope has keys outside the staging contract: {sorted(extra)}")
    return envelope


def write_batch(
    out_dir: Path,
    judged: Sequence[JudgedPair],
    *,
    batch_id: str,
    split_hash: str,
    labeler_meta: Mapping[str, Any],
    manifest_extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write staged/ and excluded/ trees plus a manifest. Nothing is promoted."""
    out_dir = Path(out_dir)
    staged_dir = out_dir / "staged"
    excluded_dir = out_dir / "excluded"
    staged_dir.mkdir(parents=True, exist_ok=True)
    excluded_dir.mkdir(parents=True, exist_ok=True)

    counts = {"pos": 0, "neg": 0, "excluded": 0}
    records: list[dict[str, Any]] = []

    for item in judged:
        label = item.label
        if label is None:
            envelope = build_envelope(
                item,
                batch_id=batch_id,
                split_hash=split_hash,
                labeler_meta=labeler_meta,
                drop_reason="judge_returned_no_verdict",
            )
            target = excluded_dir / f"{item.pair_key}.json"
            counts["excluded"] += 1
        else:
            envelope = build_envelope(
                item,
                batch_id=batch_id,
                split_hash=split_hash,
                labeler_meta=labeler_meta,
            )
            target = staged_dir / f"{item.pair_key}.json"
            counts[label] += 1

        target.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
        records.append(
            {
                "pair_key": item.pair_key,
                "seeker_id": item.seeker_id,
                "candidate_id": item.candidate_id,
                "query_key": item.query_key,
                "label": label,
                "path": str(target.relative_to(out_dir)),
            }
        )

    manifest = {
        "batch_id": batch_id,
        "split_hash": split_hash,
        "counts": counts,
        "labeler": dict(labeler_meta),
        "promoted": False,
        "promotion_note": (
            "Labels are a model's opinion, not real accept/decline outcomes. "
            "Nothing here is promoted into data/dataset_*.json."
        ),
        "records": records,
        **(dict(manifest_extra) if manifest_extra else {}),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def label_balance(judged: Sequence[JudgedPair]) -> dict[str, Any]:
    """Positive/negative split and graph density, both reported against real data.

    Real pairs run 0.673 edges per node and a balanced 100/100 split; this batch
    is expected to diverge on both, and the divergence is measured rather than
    assumed.
    """
    pos = sum(1 for j in judged if j.label == LABEL_POSITIVE)
    neg = sum(1 for j in judged if j.label == LABEL_NEGATIVE)
    labeled = [j for j in judged if j.label is not None]
    nodes = {j.seeker_id for j in labeled} | {j.candidate_id for j in labeled}
    return {
        "positive": pos,
        "negative": neg,
        "positive_frac": round(pos / max(pos + neg, 1), 4),
        "nodes": len(nodes),
        "edges": len(labeled),
        "edges_per_node": round(len(labeled) / max(len(nodes), 1), 4),
        "real_reference": {"edges_per_node": 0.673, "positive_frac": 0.5},
    }
