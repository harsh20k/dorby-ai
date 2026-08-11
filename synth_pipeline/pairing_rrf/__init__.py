"""Dual-channel retrieval + LLM-judge labeling for standalone synthetic profiles.

A sibling to ``synth_pipeline/pairing`` rather than a replacement. Both turn a
pool of unlabeled profiles into labeled pos/neg pairs; they differ in how the
label is produced, and that difference is the point.

``pairing`` labels with the TF-IDF + Voyage-nano fusion scorer. On its first real
batch that turned out to be near-circular: ``select.py`` ranks candidates by
TF-IDF and a TF-IDF-heavy scorer then grades that ranking, so plain query cosine
predicted the assigned label at 0.868 AUC — the label was largely lexical overlap.

``pairing_rrf`` breaks that loop by construction. Retrieval and labeling come
from different model families: an open-weight embedding model plus BM25 propose
candidates, and ``google/gemini-3.1-flash-lite`` — measured at 0.6358 pair AUC
and the project's best hard-negative AUC at 0.6466 — decides. No scorer grades
its own ranking.

Labels remain a model's opinion, not real accept/decline outcomes. Batches stay
in their own ``artifacts/pairing_rrf/<batch_id>/`` namespace and nothing is
promoted into ``data/dataset_*.json``.
"""

from __future__ import annotations

__all__ = [
    "embed",
    "fuse",
    "judge",
    "label",
    "prompt_hub",
    "query_gen",
    "recall",
    "run",
    "sections",
    "store",
]
