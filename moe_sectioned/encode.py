"""Turn section text and candidate text into vectors, under a swappable backend.

Two backends, and the split matters for cost:

``tfidf``  Free, local, instant. Uses ``baselines.tfidf.encode.TfidfEncoder``
           unchanged so the lexical channel matches every other result in the
           repo. This is what the whole pipeline runs on by default.

``qwen3``  Qwen3-Embedding-8B, the only model measured to beat Voyage-4-large on
           this task. Needs a GPU, so vectors are produced ahead of time by
           ``modal_encode.py`` (a paid run) and merely *read* here.

The reason a backend abstraction exists at all: the last experiment had to throw
away its embedding channel entirely because the real pairs were encoded with
voyage-nano and the synthetic ones with Qwen3, and a cosine from two different
models is not the same feature. Putting the encoder behind one interface makes
"both populations, one model" checkable rather than assumed — see
``assert_same_space``.

**Never call ``TfidfEncoder.encode()`` here.** Its disk cache keys on
``(texts, max_features, ngram_range)`` and *not* on the fitted vocabulary, so
encoding the same text under two different fits silently returns the first fit's
vectors. That bug merged two arms of the previous experiment into one and was
only caught because both reported byte-identical AUCs. We fit with
``TfidfEncoder`` (to inherit its exact parameters) and call
``vectorizer.transform`` directly, which touches no cache.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from .config import ENCODER_QWEN3, ENCODER_TFIDF


class Encoder(Protocol):
    name: str

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """(N, D) L2-normalized rows, so a dot product is a cosine."""


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


@dataclass
class TfidfBackend:
    """Fit-on-train, transform-everything. Free and deterministic."""

    name: str = ENCODER_TFIDF
    _vectorizer: object | None = None

    def fit(self, texts: Sequence[str]) -> "TfidfBackend":
        """``texts`` must come from the training population only.

        Fitting on anything the model is later evaluated against leaks the
        evaluation vocabulary into training, which is the quiet version of the
        mistake this repo keeps guarding against.
        """
        from baselines.tfidf.encode import TfidfEncoder

        enc = TfidfEncoder()
        enc.fit(list(texts))
        assert enc.vectorizer is not None
        self._vectorizer = enc.vectorizer
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self._vectorizer is None:
            raise RuntimeError("call fit() before encode()")
        return _l2(self._vectorizer.transform(list(texts)).toarray().astype(np.float32))


@dataclass
class Qwen3Backend:
    """Reads vectors precomputed by ``modal_encode.py``. Never calls a GPU itself.

    Lookup is by SHA-256 of the exact text, so a cache built for one revision of
    the section splitter cannot silently serve a different one — a changed
    splitter produces different text, which misses, which raises rather than
    returning a stale vector.
    """

    embedding_dir: Path
    name: str = ENCODER_QWEN3
    _table: dict[str, int] | None = None
    _matrix: np.ndarray | None = None

    def _load(self) -> None:
        if self._table is not None:
            return
        idx = self.embedding_dir / "index.json"
        vecs = self.embedding_dir / "vectors.npy"
        if not idx.exists() or not vecs.exists():
            raise FileNotFoundError(
                f"no Qwen3 embeddings under {self.embedding_dir}. Produce them with "
                "`modal run moe_sectioned/modal_encode.py` (a paid GPU run) or use "
                "--encoder tfidf, which is free and runs locally."
            )
        self._table = json.loads(idx.read_text())["hash_to_row"]
        self._matrix = np.load(vecs)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self._load()
        assert self._table is not None and self._matrix is not None
        rows, missing = [], 0
        for t in texts:
            h = text_hash(t)
            r = self._table.get(h)
            if r is None:
                missing += 1
                rows.append(-1)
            else:
                rows.append(r)
        if missing:
            raise KeyError(
                f"{missing}/{len(texts)} texts have no cached Qwen3 vector. The "
                "embedding cache is keyed on exact text, so this means the section "
                "splitter or text packing changed since the encode run — re-encode "
                "rather than proceeding with partial vectors."
            )
        return _l2(self._matrix[np.array(rows)].astype(np.float32))


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def assert_same_space(*encoders: Encoder) -> None:
    """Both populations must be encoded by the same backend.

    The previous experiment lost its entire embedding channel to exactly this
    mismatch. Cheap to check, expensive to miss.
    """
    names = {e.name for e in encoders}
    if len(names) > 1:
        raise AssertionError(
            f"populations encoded by different backends {sorted(names)} — a cosine "
            "from two different models is not the same feature, so the model would "
            "learn a threshold in one space and apply it in another"
        )
