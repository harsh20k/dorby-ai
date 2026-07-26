"""Weighted reciprocal-rank fusion over the two recall channels, then the cut.

Fusing by *rank* rather than by score is the version this repo's own numbers
support. On identical inputs over the matched 69-pair holdout, RRF and score
fusion produced the same pair AUC (0.6397) but RRF ranked better — MRR 0.4043
against 0.3665 (``artifacts/hybrid_tfidf_voyage_holdout_{alpha_rrf,alpha_score_fusion}``).
``rrf_k=60`` matches what those runs used.

The dense channel carries double weight because the same holdout measured each
channel alone: dense MRR 0.4610 against lexical 0.2939, a 57% edge on the metric
that governs recall. (Note the split — lexical is the *better* pair scorer,
0.6474 against 0.5793 AUC. It is better at judging a pair handed to it and worse
at finding one, and retrieval is the job here.) The 2:1 ratio tracks the measured
0.4610/0.2939 and is carried over deliberately: this batch has no ground truth,
so tuning the weight on it would only fit the judge's opinions.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from synth_pipeline.pairing_rrf.recall import QueryRecall

RRF_K = 60
DENSE_WEIGHT = 2.0
LEXICAL_WEIGHT = 1.0


@dataclass
class FusedCandidate:
    """One candidate's standing after fusion, with both channels' evidence kept."""

    candidate_id: str
    rrf_score: float
    dense_rank: int | None = None
    dense_score: float | None = None
    lexical_rank: int | None = None
    lexical_score: float | None = None

    @property
    def found_by_both(self) -> bool:
        return self.dense_rank is not None and self.lexical_rank is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "rrf_score": round(self.rrf_score, 6),
            "dense_rank": self.dense_rank,
            "dense_score": None if self.dense_score is None else round(self.dense_score, 6),
            "lexical_rank": self.lexical_rank,
            "lexical_score": None if self.lexical_score is None else round(self.lexical_score, 6),
            "found_by_both": self.found_by_both,
        }


@dataclass
class Shortlist:
    """What actually goes to the judge for one query."""

    seeker_id: str
    section_index: int
    query_key: str
    query_text: str
    candidates: list[FusedCandidate] = field(default_factory=list)
    dropped: list[dict[str, Any]] = field(default_factory=list)


def rrf_fuse(
    recall: QueryRecall,
    *,
    dense_weight: float = DENSE_WEIGHT,
    lexical_weight: float = LEXICAL_WEIGHT,
    rrf_k: int = RRF_K,
) -> list[FusedCandidate]:
    """Weighted RRF. A candidate found by both channels accrues from both."""
    acc: dict[str, FusedCandidate] = {}

    def contribute(hits, weight: float, which: str) -> None:
        for rank0, hit in enumerate(hits):
            rank = rank0 + 1
            entry = acc.get(hit.candidate_id)
            if entry is None:
                entry = FusedCandidate(candidate_id=hit.candidate_id, rrf_score=0.0)
                acc[hit.candidate_id] = entry
            entry.rrf_score += weight / (rrf_k + rank)
            if which == "dense":
                entry.dense_rank, entry.dense_score = rank, hit.similarity
            else:
                entry.lexical_rank, entry.lexical_score = rank, hit.similarity

    contribute(recall.dense.hits, dense_weight, "dense")
    contribute(recall.lexical.hits, lexical_weight, "lexical")
    return sorted(acc.values(), key=lambda c: -c.rrf_score)


def _fill_dense_scores(
    fused: Sequence[FusedCandidate],
    *,
    seeker_vectors: np.ndarray,
    candidate_index: dict[str, int],
    candidate_matrix: np.ndarray,
) -> None:
    """Give lexical-only candidates a dense similarity too, so the floor is fair.

    A candidate BM25 surfaced but dense retrieval did not has ``dense_score``
    unset, and would otherwise sail past any similarity floor untested.
    """
    if not len(seeker_vectors):
        return
    for cand in fused:
        if cand.dense_score is not None:
            continue
        row = candidate_index.get(cand.candidate_id)
        if row is None:
            continue
        sims = seeker_vectors @ candidate_matrix[row]
        cand.dense_score = float(np.max(sims))


def build_shortlist(
    recall: QueryRecall,
    *,
    seeker_vectors: np.ndarray,
    candidate_index: dict[str, int],
    candidate_matrix: np.ndarray,
    top_k: int = 5,
    min_dense_similarity: float | None = None,
    dense_weight: float = DENSE_WEIGHT,
    lexical_weight: float = LEXICAL_WEIGHT,
    rrf_k: int = RRF_K,
) -> Shortlist:
    """Fuse, apply the similarity floor, then keep the top ``top_k``.

    ``min_dense_similarity`` is off by default and that is deliberate. A floor
    is only meaningful as an absolute number, and an absolute number for this
    model would have to be calibrated against real pairs — picking one from the
    synthetic batch's own distribution repeats the mistake documented for
    ``--label-mode absolute`` in the earlier pairing pipeline, where synthetic
    scores did not overlap the real range at all.
    """
    fused = rrf_fuse(
        recall,
        dense_weight=dense_weight,
        lexical_weight=lexical_weight,
        rrf_k=rrf_k,
    )
    _fill_dense_scores(
        fused,
        seeker_vectors=seeker_vectors,
        candidate_index=candidate_index,
        candidate_matrix=candidate_matrix,
    )

    dropped: list[dict[str, Any]] = []
    kept: list[FusedCandidate] = []
    for cand in fused:
        if (
            min_dense_similarity is not None
            and cand.dense_score is not None
            and cand.dense_score < min_dense_similarity
        ):
            dropped.append({**cand.to_dict(), "drop_reason": "below_similarity_floor"})
            continue
        kept.append(cand)

    for cand in kept[top_k:]:
        dropped.append({**cand.to_dict(), "drop_reason": "below_top_k"})

    return Shortlist(
        seeker_id=recall.target.contact_id,
        section_index=recall.target.section_index,
        query_key=recall.target.key,
        query_text=recall.query_text,
        candidates=kept[:top_k],
        dropped=dropped,
    )


def deduplicate_pairs(shortlists: Sequence[Shortlist]) -> list[Shortlist]:
    """Enforce global ``(seeker, candidate)`` uniqueness, keeping the best query.

    A seeker with several ``lookingFor`` sections issues several queries, and the
    same candidate can be shortlisted by more than one of them. Left alone that
    produces duplicate pairs — and, once judged independently, *contradictory*
    ones: the first run of this pipeline labeled 25 pairs both ``pos`` and ``neg``
    under different queries from the same seeker.

    Defensible in the abstract — a person really can be a good match for one ask
    and a poor one for another — but the pair schema carries no query identity,
    and ``promote.py`` dedups on ``(userContactId, matchContactId)``, so a
    downstream consumer would silently keep whichever copy it saw first. The
    existing ``pairing/select.py`` enforces the same invariant for this reason.

    The surviving copy is the one whose query retrieved the candidate most
    strongly (highest fused score), matching the tie-break used elsewhere.
    """
    best: dict[tuple[str, str], tuple[int, FusedCandidate]] = {}
    for i, sl in enumerate(shortlists):
        for cand in sl.candidates:
            key = (sl.seeker_id, cand.candidate_id)
            current = best.get(key)
            if current is None or cand.rrf_score > current[1].rrf_score:
                best[key] = (i, cand)

    keep = {(i, cand.candidate_id) for i, cand in best.values()}
    out: list[Shortlist] = []
    for i, sl in enumerate(shortlists):
        kept, cut = [], list(sl.dropped)
        for cand in sl.candidates:
            if (i, cand.candidate_id) in keep:
                kept.append(cand)
            else:
                cut.append({**cand.to_dict(), "drop_reason": "duplicate_seeker_candidate"})
        out.append(
            Shortlist(
                seeker_id=sl.seeker_id,
                section_index=sl.section_index,
                query_key=sl.query_key,
                query_text=sl.query_text,
                candidates=kept,
                dropped=cut,
            )
        )
    return out


def apply_seeker_budget(
    shortlists: Sequence[Shortlist], *, max_pairs_per_seeker: int | None
) -> list[Shortlist]:
    """Cap judged pairs per seeker across all of their sections.

    Without this a seeker with four ``lookingFor`` sections quietly consumes
    four times the judge budget of a single-ask seeker. Trimming is
    most-confident-first by fused score, and everything cut is recorded with a
    reason rather than vanishing.
    """
    if max_pairs_per_seeker is None:
        return list(shortlists)

    by_seeker: dict[str, list[tuple[int, FusedCandidate]]] = defaultdict(list)
    for i, sl in enumerate(shortlists):
        for cand in sl.candidates:
            by_seeker[sl.seeker_id].append((i, cand))

    keep: set[tuple[int, str]] = set()
    for seeker_id, entries in by_seeker.items():
        entries.sort(key=lambda pair: -pair[1].rrf_score)
        for idx, cand in entries[:max_pairs_per_seeker]:
            keep.add((idx, cand.candidate_id))

    out: list[Shortlist] = []
    for i, sl in enumerate(shortlists):
        kept, cut = [], list(sl.dropped)
        for cand in sl.candidates:
            if (i, cand.candidate_id) in keep:
                kept.append(cand)
            else:
                cut.append({**cand.to_dict(), "drop_reason": "seeker_budget"})
        out.append(
            Shortlist(
                seeker_id=sl.seeker_id,
                section_index=sl.section_index,
                query_key=sl.query_key,
                query_text=sl.query_text,
                candidates=kept,
                dropped=cut,
            )
        )
    return out
