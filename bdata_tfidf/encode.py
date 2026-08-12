"""TF-IDF encoder — copied from baselines/tfidf/encode.py (isolation rule).

Do not edit baselines/tfidf/. Keep this file numerically aligned with the
source; tests/test_bdata_tfidf_data.py checks cosine_scores drift.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def pick_device() -> str:
    return "cpu"


def _cache_key(texts: Sequence[str], max_features: int, ngram_range: tuple[int, int]) -> str:
    h = hashlib.sha256()
    h.update(str(max_features).encode())
    h.update(b"\0")
    h.update(str(ngram_range).encode())
    h.update(b"\0")
    for text in texts:
        h.update(text.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:32]


class TfidfEncoder:
    """Fit once on the train corpus, then encode() any subset."""

    def __init__(
        self,
        max_features: int = 20000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 1,
        cache_dir: Path | None = None,
    ) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.cache_dir = Path(cache_dir) if cache_dir else Path("artifacts/bdata_tfidf")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.vectorizer: TfidfVectorizer | None = None

    def fit(self, corpus_texts: Sequence[str]) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
        )
        self.vectorizer.fit(corpus_texts)

    def encode(
        self,
        texts: Sequence[str],
        cache_name: str | None = None,
        **_ignored,
    ) -> np.ndarray:
        if self.vectorizer is None:
            raise RuntimeError("call fit() with the train corpus before encode()")
        texts_list = list(texts)
        if not texts_list:
            return np.zeros(
                (0, len(self.vectorizer.get_feature_names_out())), dtype=np.float32
            )

        key = cache_name or _cache_key(texts_list, self.max_features, self.ngram_range)
        cache_path = self.cache_dir / f"emb_{key}.npy"
        if cache_path.exists():
            return np.load(cache_path)

        matrix = self.vectorizer.transform(texts_list).toarray().astype(np.float32)
        np.save(cache_path, matrix)
        return matrix


def cosine_scores(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine for L2-normalized vectors (elementwise product sum)."""
    if a.ndim == 1:
        a = a[None, :]
    if b.ndim == 1:
        b = b[None, :]
    return np.sum(a * b, axis=-1)
