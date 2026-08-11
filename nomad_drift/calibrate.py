"""Fit the query/profile blend alpha on rrf_003 (1000-profile synthetic batch).

Reuses ``query_weighted.text`` (seeker/candidate text builders) and
``query_weighted.eval.combine`` (the alpha blend itself) unmodified — the only
new code here is loading rrf_003's pair records and defining the two
calibration metrics (see module docstring in ``nomad_drift/__init__.py``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from baselines.metrics import pair_metrics
from baselines.voyage_nano.encode import l2_normalize
from query_weighted import text as qtext

RRF_BATCH_DEFAULT = Path("artifacts/pairing_rrf/rrf_003")

# 0.0 and 1.0 included deliberately (unlike query_weighted's own grid): they
# reproduce profile_only / query_only exactly and anchor the sweep's endpoints.
ALPHAS: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


@dataclass(frozen=True)
class RrfRecord:
    pair_key: str
    seeker_id: str
    candidate_id: str
    query_key: str
    label: str
    user_contact_file: dict[str, Any]
    match_contact_file: dict[str, Any]
    search_query: str


def load_rrf_records(batch_dir: Path = RRF_BATCH_DEFAULT) -> list[RrfRecord]:
    """Every staged pair in a pairing_rrf batch, as flat records.

    Reads ``manifest.json``'s record list (pair_key/seeker_id/candidate_id/
    query_key/label/path) and follows each ``path`` into the staged pair JSON
    for the actual profile text — same schema as a real pair
    (``userContactFile``/``matchContactFile``/``searchQuery``), so
    ``query_weighted.text`` builders apply unchanged.
    """
    manifest = json.loads((batch_dir / "manifest.json").read_text())
    records: list[RrfRecord] = []
    for rec in manifest["records"]:
        staged = json.loads((batch_dir / rec["path"]).read_text())
        pair = staged["pair"]
        records.append(
            RrfRecord(
                pair_key=rec["pair_key"],
                seeker_id=rec["seeker_id"],
                candidate_id=rec["candidate_id"],
                query_key=rec["query_key"],
                label=rec["label"],
                user_contact_file=pair["userContactFile"],
                match_contact_file=pair["matchContactFile"],
                search_query=pair["searchQuery"],
            )
        )
    return records


@dataclass(frozen=True)
class Encoded:
    """Deduped seeker-profile / query / candidate vectors, keyed for lookup."""

    records: list[RrfRecord]
    profile_by_seeker: dict[str, np.ndarray]
    query_by_key: dict[str, np.ndarray]
    candidate_by_id: dict[str, np.ndarray]


def encode_calibration_set(
    encoder: Any, records: list[RrfRecord], batch_size: int = 4
) -> Encoded:
    """One encode pass over rrf_003: dedupe seeker profiles, queries, candidates.

    923 unique profiles play both seeker and candidate roles across the batch,
    but the two encodes use different Voyage prompts (``role="query"`` for the
    seeker side, ``role="document"`` for the candidate side, matching every
    other baseline in this project), so profile-as-seeker and profile-as-candidate
    vectors are deliberately kept separate even for the same person.
    """
    profile_texts: dict[str, str] = {}
    query_texts: dict[str, str] = {}
    candidate_texts: dict[str, str] = {}
    for r in records:
        profile_texts.setdefault(r.seeker_id, qtext.profile_only(r.user_contact_file))
        query_texts.setdefault(
            r.query_key, qtext.query_only(r.user_contact_file, r.search_query)
        )
        candidate_texts.setdefault(
            r.candidate_id, qtext.candidate_to_text(r.match_contact_file)
        )

    seeker_ids = list(profile_texts)
    query_keys = list(query_texts)
    cand_ids = list(candidate_texts)

    print(f"  encoding {len(seeker_ids)} unique seeker profiles")
    profile_mat = encoder.encode(
        [profile_texts[s] for s in seeker_ids],
        role="query",
        batch_size=batch_size,
        show_progress=False,
    )
    print(f"  encoding {len(query_keys)} unique queries")
    query_mat = encoder.encode(
        [query_texts[q] for q in query_keys],
        role="query",
        batch_size=batch_size,
        show_progress=False,
    )
    print(f"  encoding {len(cand_ids)} unique candidates")
    cand_mat = encoder.encode(
        [candidate_texts[c] for c in cand_ids],
        role="document",
        batch_size=batch_size,
        show_progress=False,
    )

    return Encoded(
        records=records,
        profile_by_seeker={s: profile_mat[i] for i, s in enumerate(seeker_ids)},
        query_by_key={q: query_mat[i] for i, q in enumerate(query_keys)},
        candidate_by_id={c: cand_mat[i] for i, c in enumerate(cand_ids)},
    )


@dataclass(frozen=True)
class AlignedArrays:
    """Encoded vectors expanded to one row per pair record, for vectorized scoring."""

    profile: np.ndarray  # (n, d)
    query: np.ndarray  # (n, d)
    candidate: np.ndarray  # (n, d)
    labels: list[str]
    query_keys: list[str]


def align(enc: Encoded) -> AlignedArrays:
    return AlignedArrays(
        profile=np.stack([enc.profile_by_seeker[r.seeker_id] for r in enc.records]),
        query=np.stack([enc.query_by_key[r.query_key] for r in enc.records]),
        candidate=np.stack([enc.candidate_by_id[r.candidate_id] for r in enc.records]),
        labels=[r.label for r in enc.records],
        query_keys=[r.query_key for r in enc.records],
    )


def combine_batch(query_vecs: np.ndarray, profile_vecs: np.ndarray, alpha: float) -> np.ndarray:
    """normalize(alpha*query + (1-alpha)*profile) — same formula as query_weighted.eval.combine."""
    return l2_normalize(alpha * query_vecs + (1.0 - alpha) * profile_vecs).astype(np.float32)


def score_alpha(arrays: AlignedArrays, alpha: float) -> dict[str, Any]:
    """Pair AUC + shortlist top-1 accuracy for one alpha, over every rrf_003 pair."""
    seeker = combine_batch(arrays.query, arrays.profile, alpha)
    scores = np.sum(seeker * arrays.candidate, axis=-1)  # both L2-normalized

    pos_mask = np.array([lab == "pos" for lab in arrays.labels])
    pos_scores = scores[pos_mask]
    neg_scores = scores[~pos_mask]
    pair = pair_metrics(pos_scores, neg_scores)

    # Shortlist top-1: within each query_key's judged candidates, does the
    # highest-scoring one carry a pos label? Only counted where both labels
    # are present for that query_key — otherwise there is nothing to rank.
    by_query: dict[str, list[tuple[str, float]]] = {}
    for qk, lab, sc in zip(arrays.query_keys, arrays.labels, scores):
        by_query.setdefault(qk, []).append((lab, float(sc)))

    correct = 0
    total = 0
    for items in by_query.values():
        labels_here = {lab for lab, _ in items}
        if "pos" not in labels_here or "neg" not in labels_here:
            continue
        total += 1
        top_label, _ = max(items, key=lambda t: t[1])
        if top_label == "pos":
            correct += 1

    return {
        "alpha": alpha,
        "pair_auc": pair["roc_auc"],
        "mean_cosine_gap": pair["mean_cosine_gap"],
        "n_pos": int(pos_mask.sum()),
        "n_neg": int((~pos_mask).sum()),
        "shortlist_top1_accuracy": (correct / total) if total else None,
        "shortlist_n_query_keys": total,
    }


def sweep(arrays: AlignedArrays, alphas: tuple[float, ...] = ALPHAS) -> list[dict[str, Any]]:
    return [score_alpha(arrays, a) for a in alphas]


def pick_best_alpha(sweep_results: list[dict[str, Any]]) -> float:
    """Best by pair AUC (corpus-free, non-circular), ties broken by shortlist accuracy."""
    ranked = sorted(
        sweep_results,
        key=lambda r: (r["pair_auc"], r["shortlist_top1_accuracy"] or 0.0),
        reverse=True,
    )
    return float(ranked[0]["alpha"])


def run_calibration(encoder: Any, batch_dir: Path = RRF_BATCH_DEFAULT, batch_size: int = 4) -> dict[str, Any]:
    records = load_rrf_records(batch_dir)
    print(f"loaded {len(records)} rrf_003 pairs")
    enc = encode_calibration_set(encoder, records, batch_size=batch_size)
    arrays = align(enc)
    results = sweep(arrays)
    best_alpha = pick_best_alpha(results)
    for r in results:
        marker = "  <-- best" if r["alpha"] == best_alpha else ""
        auc = r["pair_auc"]
        acc = r["shortlist_top1_accuracy"]
        acc_s = f"{acc:.4f}" if acc is not None else "n/a"
        print(f"  alpha={r['alpha']:.1f}  pairAUC={auc:.4f}  shortlist_top1={acc_s}{marker}")
    return {
        "batch_dir": str(batch_dir),
        "n_records": len(records),
        "n_unique_seekers": len({r.seeker_id for r in records}),
        "n_unique_candidates": len({r.candidate_id for r in records}),
        "n_unique_query_keys": len({r.query_key for r in records}),
        "sweep": results,
        "best_alpha": best_alpha,
    }
