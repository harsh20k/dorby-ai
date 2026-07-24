"""Pick which candidates each (seeker, query) gets paired against.

Pure — no LLM, no network, deterministic under a fixed seed.

Two decisions worth stating, because both are easy to get subtly wrong:

1. Candidates are ranked by similarity to the QUERY, not to the seeker profile.
   "Topically adjacent" only means anything relative to what is being asked for.

2. Sampling is log-spaced from the TOP of the ranking and never reaches the tail.
   The tail is trivially-unrelated easy negatives, and `data/synthetic/strategy.md`
   established those don't move hard-negative AUC — they'd be spent budget for a
   metric that doesn't move.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from baselines.bert_frozen.text import candidate_to_text
from synth_pipeline.pairing.profiles import SynthProfile

Band = Literal["top", "near", "mid"]


@dataclass(frozen=True)
class Candidate:
    seeker: SynthProfile
    candidate: SynthProfile
    query: str
    query_index: int
    cosine: float
    rank: int
    band: Band
    same_archetype: bool


def _log_spaced_ranks(n_available: int, k: int) -> list[int]:
    """0-indexed ranks spread geometrically from the top: 0, 1, 3, 7, 14, ..."""
    ranks: list[int] = []
    step = 1
    r = 0
    while len(ranks) < k and r < n_available:
        ranks.append(r)
        r += step
        step *= 2
    # If the geometric walk ran out of room, backfill from the top.
    for r in range(n_available):
        if len(ranks) >= k:
            break
        if r not in ranks:
            ranks.append(r)
    return sorted(ranks[:k])


def _band_for_rank(rank: int) -> Band:
    if rank == 0:
        return "top"
    if rank <= 3:
        return "near"
    return "mid"


def select_candidates(
    profiles: list[SynthProfile],
    queries: dict[str, list[str]],
    *,
    k_per_query: int = 5,
    max_cosine: float = 0.90,
) -> list[Candidate]:
    """Rank all other profiles against each query, take a log-spaced top band."""
    if len(profiles) < 2:
        return []

    by_id = {p.contact_id: p for p in profiles}
    cand_texts = [candidate_to_text(p.profile) for p in profiles]
    all_queries = [q for qs in queries.values() for q in qs]

    # Fit one shared vocabulary over candidates and queries so the two sides live
    # in the same IDF space (same requirement as baselines/tfidf/encode.py).
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=1)
    vec.fit(cand_texts + all_queries)
    cand_mat = vec.transform(cand_texts).toarray().astype(np.float32)

    out: list[Candidate] = []
    seen: set[tuple[str, str]] = set()

    for seeker in profiles:
        for q_idx, query in enumerate(queries.get(seeker.contact_id, [])):
            q_vec = vec.transform([query]).toarray().astype(np.float32)[0]
            sims = cand_mat @ q_vec  # both L2-normalized by TfidfVectorizer

            order = [
                i
                for i in np.argsort(-sims)
                if profiles[i].contact_id != seeker.contact_id
                and float(sims[i]) <= max_cosine
            ]
            if not order:
                continue

            for rank in _log_spaced_ranks(len(order), k_per_query):
                cand = profiles[order[rank]]
                key = (seeker.contact_id, cand.contact_id)
                # The pair schema has no query identity, and promote.py dedups on
                # (userContactId, matchContactId) — so the same two people cannot
                # carry two labels. Enforce it here rather than losing pairs later.
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    Candidate(
                        seeker=seeker,
                        candidate=cand,
                        query=query,
                        query_index=q_idx,
                        cosine=float(sims[order[rank]]),
                        rank=rank,
                        band=_band_for_rank(rank),
                        same_archetype=seeker.archetype == cand.archetype,
                    )
                )

    assert len({(c.seeker.contact_id, c.candidate.contact_id) for c in out}) == len(out)
    assert all(c.seeker.contact_id != c.candidate.contact_id for c in out)
    assert all(c.candidate.contact_id in by_id for c in out)
    return out
