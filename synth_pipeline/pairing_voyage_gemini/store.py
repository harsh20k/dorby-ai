"""Local vector store — Chroma, built from the persisted ``.npy`` files.

The arrays written by ``embed.py`` are the source of truth; this module is a
queryable layer over them, never the only copy. ``build_chroma`` is idempotent
and rebuildable from disk at any time without touching a GPU.

An exact NumPy store implementing the same interface is included and used as the
fallback when ``chromadb`` is unavailable. At this batch size it is not a
downgrade: 57 candidate vectors is a single matrix multiply, microseconds, and
exactly correct with no ANN approximation. Chroma earns its place as the shape
that keeps working when the candidate pool is thousands of profiles, and because
it carries the per-vector metadata (which seeker, which section) that the N+1
seeker layout needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

COLLECTION = "candidates"


@dataclass
class Hit:
    """One retrieved candidate."""

    candidate_id: str
    similarity: float  # cosine, in [-1, 1]; higher is closer


class VectorStore(Protocol):
    def query(self, vectors: np.ndarray, k: int) -> list[list[Hit]]: ...
    @property
    def size(self) -> int: ...


class ExactStore:
    """Brute-force cosine over an in-memory matrix. Exact, no index."""

    backend = "exact-numpy"

    def __init__(self, ids: Sequence[str], matrix: np.ndarray) -> None:
        self.ids = list(ids)
        self.matrix = np.asarray(matrix, dtype=np.float32)

    @property
    def size(self) -> int:
        return len(self.ids)

    def query(self, vectors: np.ndarray, k: int) -> list[list[Hit]]:
        if not self.ids:
            return [[] for _ in range(len(vectors))]
        sims = np.asarray(vectors, dtype=np.float32) @ self.matrix.T
        k = min(k, len(self.ids))
        out: list[list[Hit]] = []
        for row in sims:
            top = np.argpartition(-row, k - 1)[:k]
            top = top[np.argsort(-row[top])]
            out.append([Hit(self.ids[i], float(row[i])) for i in top])
        return out


class ChromaStore:
    """Persistent local Chroma collection, cosine space."""

    backend = "chroma"

    def __init__(self, collection: Any) -> None:
        self.collection = collection

    @property
    def size(self) -> int:
        return int(self.collection.count())

    def query(self, vectors: np.ndarray, k: int) -> list[list[Hit]]:
        k = min(k, self.size)
        if k <= 0:
            return [[] for _ in range(len(vectors))]
        res = self.collection.query(
            query_embeddings=np.asarray(vectors, dtype=np.float32).tolist(),
            n_results=k,
            include=["distances"],
        )
        out: list[list[Hit]] = []
        for ids, dists in zip(res["ids"], res["distances"]):
            # Chroma cosine distance is 1 - cosine similarity.
            out.append([Hit(i, 1.0 - float(d)) for i, d in zip(ids, dists)])
        return out


def build_chroma(
    chroma_dir: Path,
    candidate_ids: Sequence[str],
    candidate_matrix: np.ndarray,
    *,
    metadatas: Sequence[dict[str, Any]] | None = None,
    collection_name: str = COLLECTION,
) -> ChromaStore:
    """Create (or replace) the candidate collection from arrays already on disk."""
    import chromadb

    chroma_dir = Path(chroma_dir)
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))

    # Rebuilt from the .npy source of truth every time, so a stale collection is
    # never silently reused.
    try:
        client.delete_collection(collection_name)
    except Exception:  # noqa: BLE001 — absent collection is the normal first-run case
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Chroma enforces a max_batch_size per collection.add() call (observed 5461
    # on this client version); every prior batch run here had far fewer than
    # that many candidates so this was never exercised until the first
    # 5000+-candidate pool. Chunk rather than raise the limit — client.get_max_batch_size()
    # is what Chroma itself uses to validate, so read it back instead of hardcoding it.
    max_batch = client.get_max_batch_size()
    ids = list(candidate_ids)
    embeddings = np.asarray(candidate_matrix, dtype=np.float32).tolist()
    meta_list = list(metadatas) if metadatas else None
    for start in range(0, len(ids), max_batch):
        end = start + max_batch
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            metadatas=meta_list[start:end] if meta_list else None,
        )
    return ChromaStore(collection)


def open_store(
    chroma_dir: Path,
    candidate_ids: Sequence[str],
    candidate_matrix: np.ndarray,
    *,
    use_chroma: bool = True,
    metadatas: Sequence[dict[str, Any]] | None = None,
) -> VectorStore:
    """Chroma when available and requested, exact NumPy otherwise.

    Falling back is loud — it changes which backend the manifest records — but
    not fatal, because the exact store returns the same answers at this scale.
    """
    if not use_chroma:
        return ExactStore(candidate_ids, candidate_matrix)
    try:
        return build_chroma(
            chroma_dir, candidate_ids, candidate_matrix, metadatas=metadatas
        )
    except ImportError:
        print(
            "chromadb not installed — falling back to the exact NumPy store. "
            "Results are identical at this scale; `pip install chromadb` to "
            "exercise the real index."
        )
        return ExactStore(candidate_ids, candidate_matrix)
