"""Two independent recall channels: dense embeddings and lexical BM25.

Run in parallel rather than as retrieve-then-rerank, and the difference is not
cosmetic. A reranker can only reorder what the dense channel already found, so a
lexically obvious match the embedder missed is unrecoverable. As its own channel,
BM25 contributes candidates of its own and the fusion decides.

Which channel sees what is deliberately asymmetric:

* **Dense** never sees the ``searchQuery``. It matches profile against profile,
  using the seeker's whole-profile vector together with the vector for the one
  ``lookingFor`` section this query was written from, and merging by best
  similarity — breadth from the first, precision from the second.
* **Lexical** sees only the ``searchQuery``, scored against candidate profile
  text. This is where the query re-enters the pipeline.

BM25 is implemented here rather than pulled in as a dependency: it is forty lines
of Okapi, and owning the tokenizer keeps it consistent with the rest of the repo.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from synth_pipeline.pairing_rrf_qwen_judge.sections import WHOLE, QueryTarget
from synth_pipeline.pairing_rrf_qwen_judge.store import Hit, VectorStore

_TOKEN_RE = re.compile(r"[a-z0-9]+")

BM25_K1 = 1.5
BM25_B = 0.75


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """Okapi BM25 over the candidate profile corpus."""

    def __init__(self, ids: Sequence[str], docs: Sequence[str]) -> None:
        self.ids = list(ids)
        self.docs = [tokenize(d) for d in docs]
        self.doc_len = np.array([len(d) for d in self.docs], dtype=np.float32)
        self.avg_len = float(self.doc_len.mean()) if len(self.doc_len) else 0.0
        self.freqs = [Counter(d) for d in self.docs]

        n_docs = len(self.docs)
        df: Counter[str] = Counter()
        for f in self.freqs:
            df.update(f.keys())
        # Standard BM25 idf with the +1 smoothing that keeps it non-negative.
        self.idf = {
            term: math.log(1.0 + (n_docs - n + 0.5) / (n + 0.5))
            for term, n in df.items()
        }

    def score(self, query: str) -> np.ndarray:
        terms = tokenize(query)
        scores = np.zeros(len(self.docs), dtype=np.float32)
        if not terms or not self.avg_len:
            return scores
        for term in terms:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, freq in enumerate(self.freqs):
                tf = freq.get(term, 0)
                if not tf:
                    continue
                denom = tf + BM25_K1 * (
                    1.0 - BM25_B + BM25_B * self.doc_len[i] / self.avg_len
                )
                scores[i] += idf * (tf * (BM25_K1 + 1.0)) / denom
        return scores

    def top_k(self, query: str, k: int) -> list[Hit]:
        scores = self.score(query)
        if not len(scores):
            return []
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        # A zero score means no query term appeared at all — not a candidate.
        return [Hit(self.ids[i], float(scores[i])) for i in top if scores[i] > 0.0]


@dataclass
class ChannelResult:
    """One channel's ranked candidates for one query."""

    name: str
    hits: list[Hit] = field(default_factory=list)

    def as_dicts(self) -> list[dict[str, Any]]:
        return [
            {"candidate_id": h.candidate_id, "score": round(h.similarity, 6), "rank": i + 1}
            for i, h in enumerate(self.hits)
        ]


@dataclass
class QueryRecall:
    """Both channels' output for a single (seeker, section, query)."""

    target: QueryTarget
    query_text: str
    dense: ChannelResult
    lexical: ChannelResult


def _merge_by_max(hit_lists: Iterable[list[Hit]], k: int) -> list[Hit]:
    """Union several ranked lists, keeping each candidate's best similarity."""
    best: dict[str, float] = {}
    for hits in hit_lists:
        for h in hits:
            prev = best.get(h.candidate_id)
            if prev is None or h.similarity > prev:
                best[h.candidate_id] = h.similarity
    ordered = sorted(best.items(), key=lambda kv: -kv[1])[:k]
    return [Hit(cid, sim) for cid, sim in ordered]


def dense_recall(
    store: VectorStore,
    seeker_rows: dict[str, int],
    seeker_matrix: np.ndarray,
    target: QueryTarget,
    *,
    k: int = 10,
) -> ChannelResult:
    """Query the store with this seeker's whole vector plus its section vector."""
    keys = [f"{target.contact_id}::whole"]
    section_key = f"{target.contact_id}::s{target.section_index}"
    if section_key in seeker_rows:
        keys.append(section_key)

    rows = [seeker_rows[key] for key in keys if key in seeker_rows]
    if not rows:
        return ChannelResult(name="dense")

    vectors = seeker_matrix[rows]
    per_vector = store.query(vectors, k)
    return ChannelResult(name="dense", hits=_merge_by_max(per_vector, k))


def lexical_recall(
    index: BM25Index, query_text: str, *, k: int = 10
) -> ChannelResult:
    return ChannelResult(name="lexical", hits=index.top_k(query_text, k))


def seeker_row_index(manifest: dict[str, Any]) -> dict[str, int]:
    """``vector key -> row`` from the persisted embedding manifest."""
    return {rec["key"]: int(rec["row"]) for rec in manifest["seeker"]}


def recall_all(
    targets: Sequence[QueryTarget],
    queries: dict[str, str],
    *,
    store: VectorStore,
    seeker_matrix: np.ndarray,
    seeker_rows: dict[str, int],
    bm25: BM25Index,
    k: int = 10,
) -> list[QueryRecall]:
    """Run both channels for every query that has text."""
    out: list[QueryRecall] = []
    for target in targets:
        query_text = queries.get(target.key, "")
        if not query_text.strip():
            continue
        out.append(
            QueryRecall(
                target=target,
                query_text=query_text,
                dense=dense_recall(
                    store, seeker_rows, seeker_matrix, target, k=k
                ),
                lexical=lexical_recall(bm25, query_text, k=k),
            )
        )
    return out


__all__ = [
    "BM25Index",
    "ChannelResult",
    "QueryRecall",
    "WHOLE",
    "dense_recall",
    "lexical_recall",
    "recall_all",
    "seeker_row_index",
    "tokenize",
]
