"""Frozen text -> vectors for the 200 real pairs, encoded once and reused.

Both arms need the same three things: seeker vectors, candidate vectors, and the
178-candidate retrieval corpus. Computing them once here keeps the two arms
scoring an identical population, and keeps the Voyage API path to a single pass
(which the content-hashed cache under ``artifacts/voyage_large/emb`` then serves
for free on re-runs).

Text serialization is imported unchanged from ``baselines.bert_frozen.text``,
the same function every published baseline row went through, so a cosine number
produced here must equal the corresponding row in
``docs/baseline-results-real200.md``. ``tests/test_bilinear_mf.py`` asserts that.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from baselines.bert_frozen.text import candidate_to_text, seeker_to_text
from eval_real_full.data import load_real_pairs

Backbone = Literal["tfidf", "voyage_large"]


def build_candidate_corpus(
    positives: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """Unique matchContactIds with first-seen matchContactFile text.

    Deliberate verbatim copy of the same function in
    ``eval_real_full/baseline_eval.py`` and each ``baselines/*/eval.py``. Copied
    rather than imported so this experiment cannot be broken by an edit to
    another one; ``tests/test_bilinear_mf.py`` pins it against
    ``eval_real_full`` by AST so the copies cannot drift.
    """
    ids: list[str] = []
    texts: list[str] = []
    seen: set[str] = set()
    for record in positives + negatives:
        mid = record["matchContactId"]
        if mid in seen:
            continue
        seen.add(mid)
        ids.append(mid)
        texts.append(candidate_to_text(record["matchContactFile"]))
    return ids, texts


@dataclass(frozen=True)
class PairFeatures:
    """All 200 real pairs, encoded, with the metadata every protocol needs.

    Rows are ordered positives-then-negatives, matching the argument order that
    ``baselines.metrics.pair_metrics`` and ``slice_metrics`` expect.
    """

    seeker_emb: np.ndarray  # (200, d)
    cand_emb: np.ndarray  # (200, d)
    labels: np.ndarray  # (200,) 1 = accepted, 0 = declined
    seeker_ids: list[str]  # (200,) userContactId, the CV grouping key
    subsets: list[str]  # (200,) "train" | "holdout"
    corpus_emb: np.ndarray  # (178, d)
    corpus_ids: list[str]
    positives: list[dict[str, Any]]
    negatives: list[dict[str, Any]]
    neg_seeker_texts: list[str]
    neg_cand_texts: list[str]
    pos_target_ids: list[str]
    real_data_hash: str
    backbone: str
    dim: int

    @property
    def n_pos(self) -> int:
        return int(self.labels.sum())

    def pos_mask(self) -> np.ndarray:
        return self.labels == 1


def _encode_tfidf(
    seeker_texts: Sequence[str],
    cand_texts: Sequence[str],
    corpus_texts: Sequence[str],
    *,
    max_features: int,
    ngram_range: tuple[int, int],
    cache_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import hashlib

    from baselines.tfidf.encode import TfidfEncoder

    # TfidfEncoder's content-hashed cache key covers the *texts being encoded*
    # but not the fitted vocabulary/IDF, so one cache dir shared across two fit
    # sets serves the first fit's vectors to the second. (Hit during this
    # experiment's build: a corrected fit set kept reproducing the old number.)
    # Namespacing the directory by the fit set closes it.
    fit_list = list(seeker_texts) + list(corpus_texts)
    fit_hash = hashlib.sha256(
        "\0".join(fit_list).encode("utf-8")
        + f"|{max_features}|{ngram_range}".encode()
    ).hexdigest()[:16]
    cache_dir = Path(cache_dir) / f"tfidf_fit_{fit_hash}"

    enc = TfidfEncoder(
        max_features=max_features, ngram_range=ngram_range, cache_dir=cache_dir
    )
    # Fit set is exactly seeker texts + the deduplicated corpus, matching
    # `eval_real_full/baseline_eval.py` and `baselines/tfidf/eval.py`. Appending
    # `cand_texts` as well looks harmless — every one of those texts is already
    # in `corpus_texts` — but IDF is a *document count*, so the duplicates shift
    # every weight and the published TF-IDF row stops reproducing (measured:
    # all-200 pair AUC 0.5572 instead of 0.5649).
    enc.fit(fit_list)
    # cache_name is left unset so the encoder falls back to its content-hashed
    # key; a fixed key silently serves stale vectors (the defect already found
    # once in synth_pipeline/pairing).
    return (
        enc.encode(list(seeker_texts)),
        enc.encode(list(cand_texts)),
        enc.encode(list(corpus_texts)),
    )


def _encode_voyage(
    seeker_texts: Sequence[str],
    cand_texts: Sequence[str],
    corpus_texts: Sequence[str],
    *,
    model_name: str,
    output_dimension: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from baselines.voyage_large.encode import VoyageLargeEncoder

    # Default cache_dir on purpose: artifacts/voyage_large/emb is keyed by
    # (text, input_type, dim), so this run reuses vectors the published
    # baselines already paid for and adds none of its own.
    enc = VoyageLargeEncoder(
        model_name=model_name, output_dimension=output_dimension
    )
    return (
        enc.encode(list(seeker_texts), input_type="query", show_progress=False),
        enc.encode(list(cand_texts), input_type="document", show_progress=False),
        enc.encode(list(corpus_texts), input_type="document", show_progress=False),
    )


def build_features(
    *,
    data_dir: Path,
    split_path: Path,
    backbone: Backbone,
    max_features: int = 20000,
    ngram_range: tuple[int, int] = (1, 2),
    voyage_model: str = "voyage-4-large",
    voyage_output_dimension: int = 1024,
    cache_dir: Path = Path("/tmp/bilinear_mf_cache"),
) -> PairFeatures:
    """Encode all 200 real pairs plus the shared candidate corpus, once."""
    ps = load_real_pairs(data_dir, split_path, subset="all", verify=True)

    positives = [p for p in ps.pairs if p.label == "pos"]
    negatives = [p for p in ps.pairs if p.label == "neg"]
    pos_records = [p.pair for p in positives]
    neg_records = [p.pair for p in negatives]

    pos_seeker_texts = [
        seeker_to_text(r["userContactFile"], r["searchQuery"]) for r in pos_records
    ]
    neg_seeker_texts = [
        seeker_to_text(r["userContactFile"], r["searchQuery"]) for r in neg_records
    ]
    pos_cand_texts = [candidate_to_text(r["matchContactFile"]) for r in pos_records]
    neg_cand_texts = [candidate_to_text(r["matchContactFile"]) for r in neg_records]
    corpus_ids, corpus_texts = build_candidate_corpus(pos_records, neg_records)

    seeker_texts = pos_seeker_texts + neg_seeker_texts
    cand_texts = pos_cand_texts + neg_cand_texts

    if backbone == "tfidf":
        seeker_emb, cand_emb, corpus_emb = _encode_tfidf(
            seeker_texts,
            cand_texts,
            corpus_texts,
            max_features=max_features,
            ngram_range=ngram_range,
            cache_dir=cache_dir,
        )
    elif backbone == "voyage_large":
        seeker_emb, cand_emb, corpus_emb = _encode_voyage(
            seeker_texts,
            cand_texts,
            corpus_texts,
            model_name=voyage_model,
            output_dimension=voyage_output_dimension,
        )
    else:
        raise ValueError(f"unknown backbone {backbone!r}")

    labels = np.concatenate(
        [np.ones(len(pos_records), dtype=np.int64), np.zeros(len(neg_records), dtype=np.int64)]
    )
    seeker_ids = [r["userContactId"] for r in pos_records + neg_records]
    subsets = [
        "holdout" if p.source == "real_holdout" else "train" for p in positives + negatives
    ]

    return PairFeatures(
        seeker_emb=np.asarray(seeker_emb, dtype=np.float32),
        cand_emb=np.asarray(cand_emb, dtype=np.float32),
        labels=labels,
        seeker_ids=seeker_ids,
        subsets=subsets,
        corpus_emb=np.asarray(corpus_emb, dtype=np.float32),
        corpus_ids=corpus_ids,
        positives=pos_records,
        negatives=neg_records,
        neg_seeker_texts=neg_seeker_texts,
        neg_cand_texts=neg_cand_texts,
        pos_target_ids=[r["matchContactId"] for r in pos_records],
        real_data_hash=ps.combined_hash,
        backbone=backbone,
        dim=int(np.asarray(seeker_emb).shape[1]),
    )
