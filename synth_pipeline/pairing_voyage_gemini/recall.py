"""Single recall channel: dense retrieval only, one vector per query.

No lexical/BM25 channel and no fusion in this batch — each query's own
embedding (positioning + searchQuery, see ``sections.py``) is queried directly
against the candidate store for its top-k, and that top-k *is* the shortlist.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from synth_pipeline.pairing_voyage_gemini.sections import QueryTarget
from synth_pipeline.pairing_voyage_gemini.store import Hit, VectorStore


@dataclass
class QueryRecall:
    target: QueryTarget
    query_text: str
    hits: list[Hit]


def seeker_row_index(manifest: dict[str, Any]) -> dict[str, int]:
    return {rec["key"]: int(rec["row"]) for rec in manifest["seeker"]}


def recall_all(
    targets: Sequence[QueryTarget],
    queries: dict[str, str],
    *,
    store: VectorStore,
    seeker_matrix: np.ndarray,
    seeker_rows: dict[str, int],
    k: int = 10,
) -> list[QueryRecall]:
    out: list[QueryRecall] = []
    for target in targets:
        query_text = queries.get(target.key, "")
        if not query_text.strip():
            continue
        row = seeker_rows.get(target.key)
        if row is None:
            continue
        vector = seeker_matrix[row : row + 1]
        hits = store.query(vector, k)[0]
        out.append(QueryRecall(target=target, query_text=query_text, hits=hits))
    return out


__all__ = ["QueryRecall", "recall_all", "seeker_row_index"]
