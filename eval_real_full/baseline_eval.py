"""Score the *frozen baselines* on the real pairs, per subset.

Why this exists separately from ``eval_real_full/eval.py``
----------------------------------------------------------
``eval.py`` scores sentence-transformers models that expose Voyage's
``encode_query``/``encode_document`` convention, via ``twotower.eval``. The
baseline packages (``baselines/tfidf``, ``baselines/bert_frozen``,
``baselines/hf_embedding``) use a different, older call shape: a per-package
encoder object plus an explicit ``run_eval`` body that builds the candidate
corpus itself. Their published numbers came through *that* path.

Reproducing those numbers on a larger population therefore requires walking the
same path, not a similar one. This module is a faithful transcription of the
body shared by ``baselines/{tfidf,hf_embedding}/eval.py::run_eval`` — same text
serialization, same corpus construction, same four metric calls in the same
order — with exactly one thing changed: which pairs it is handed.

Nothing under ``baselines/`` is modified. Every encoder and every metric is
imported and called unchanged, so a number produced here is directly comparable
to the corresponding row in ``docs/baseline-results-holdout.md`` — and
``--subset holdout`` must reproduce it digit-for-digit, which
``tests/test_eval_real_full_baselines.py`` asserts.

The population is the point
---------------------------
The 69-pair holdout has now reversed a conclusion three times in this project,
most recently the "Qwen3-Embedding-8B beats Voyage-4-large" headline. Every row
in the holdout table rests on 29 positive queries. This scores them on all 200
real pairs (100 positive queries, 178-candidate corpus) instead.

Corpus-size caveat: as in ``eval.py``, retrieval metrics are comparable
*between models on one subset*, never *between subsets* — a bigger candidate
pool is strictly harder. ``n_candidates`` is recorded for that reason.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from baselines.bert_frozen.text import candidate_to_text, seeker_to_text
from baselines.hf_embedding.encode import cosine_scores
from baselines.metrics import pair_metrics, retrieval_metrics, slice_metrics

from eval_real_full.data import Subset, load_real_pairs
from eval_real_full.eval import write_metrics

__all__ = [
    "build_candidate_corpus",
    "library_versions",
    "run_baseline_eval",
    "split_by_label",
    "write_metrics",
]

Kind = Literal["tfidf", "bert", "hf"]


def library_versions(modules: Sequence[str]) -> dict[str, str | None]:
    """Version provenance for libraries that can move a metric (e.g. sklearn's
    TF-IDF vocabulary — see ``eval_real_full/modal_baseline_eval.py``'s
    scikit-learn pin). ``None`` for a module not installed in the current
    environment, which is expected for some (e.g. no ``torch`` on a CPU-only run).
    """
    import importlib

    out: dict[str, str | None] = {}
    for mod in modules:
        try:
            out[mod] = importlib.import_module(mod).__version__
        except Exception:
            out[mod] = None
    return out


DEFAULT_SUBSETS: tuple[Subset, ...] = ("all", "train", "holdout")
_SOURCE_BY_SUBSET = {"train": "real_train", "holdout": "real_holdout"}


def split_by_label(pairs) -> tuple[list[dict], list[dict]]:
    """``LabeledPair``s back into the raw (positives, negatives) the baselines take."""
    positives = [p.pair for p in pairs if p.label == "pos"]
    negatives = [p.pair for p in pairs if p.label == "neg"]
    return positives, negatives


def build_candidate_corpus(
    positives: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """Unique matchContactIds with first-seen matchContactFile text.

    Deliberate verbatim copy of the identical function in
    ``baselines/{tfidf,hf_embedding,bert_frozen}/eval.py`` — importing one
    package's private copy here would silently tie this module to that one
    baseline. ``tests/test_eval_real_full_baselines.py`` pins it against all
    three by AST so the copies cannot drift.
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


class _EncoderAdapter:
    """Uniform ``encode(texts, role)`` over the three baselines' encoders.

    Only ``hf`` distinguishes query from document (asymmetric prompts); tfidf
    and bert encode both sides identically, exactly as their own eval.py does.
    ``cache_name`` is deliberately left ``None`` so every encoder falls back to
    its *content-hashed* key. A fixed key would silently serve one subset's
    embeddings to another — the stale-cache defect already found once in
    ``synth_pipeline/pairing``.
    """

    def __init__(self, kind: Kind, encoder, batch_size: int) -> None:
        self.kind = kind
        self.encoder = encoder
        self.batch_size = batch_size

    def encode(self, texts: Sequence[str], role: Literal["query", "document"]):
        if self.kind == "hf":
            return self.encoder.encode(
                texts, role=role, batch_size=self.batch_size, show_progress=False
            )
        if self.kind == "bert":
            return self.encoder.encode(
                texts, batch_size=self.batch_size, show_progress=False
            )
        return self.encoder.encode(texts)  # tfidf: no batching, no role


def _build_encoder(
    kind: Kind,
    *,
    model_name: str | None,
    device: str,
    max_length: int,
    truncate_dim: int | None,
    dtype: str,
    cache_dir: Path,
    max_features: int,
    ngram_range: tuple[int, int],
    fit_texts: Sequence[str],
):
    if kind == "tfidf":
        from baselines.tfidf.encode import TfidfEncoder

        enc = TfidfEncoder(
            max_features=max_features, ngram_range=ngram_range, cache_dir=cache_dir
        )
        # TF-IDF's vocabulary/IDF is corpus-dependent, so fit() must see every
        # text that will be encoded — and must be refit per subset, which is
        # what baselines/tfidf/eval.py --holdout-only does.
        enc.fit(list(fit_texts))
        return enc
    if kind == "bert":
        import torch

        from baselines.bert_frozen.encode import FrozenBertEncoder

        return FrozenBertEncoder(
            model_name=model_name or "bert-base-uncased",
            device=torch.device(device),
            max_length=max_length,
            cache_dir=cache_dir,
        )
    if kind == "hf":
        from baselines.hf_embedding.encode import get_encoder_class

        if not model_name:
            raise ValueError("kind='hf' requires --model")
        return get_encoder_class(model_name)(
            model_name=model_name,
            device=device,
            max_length=max_length,
            truncate_dim=truncate_dim,
            dtype=dtype,
            cache_dir=cache_dir,
        )
    raise ValueError(f"unknown kind {kind!r}")


def _score(
    *,
    label: str,
    subset: str,
    positives: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
    pos_seeker_emb: np.ndarray,
    neg_seeker_emb: np.ndarray,
    pos_cand_emb: np.ndarray,
    neg_cand_emb: np.ndarray,
    corpus_ids: list[str],
    corpus_emb: np.ndarray,
    neg_seeker_texts: list[str],
    neg_cand_texts: list[str],
    combined_hash: str,
) -> dict[str, Any]:
    """Shared scoring tail: cosine + the four metric calls, identical for every kind."""
    # Every encoder here returns L2-normalized rows, so the shared dot-product
    # cosine applies — same as each baseline's own eval.py.
    pos_scores = cosine_scores(pos_seeker_emb, pos_cand_emb)
    neg_scores = cosine_scores(neg_seeker_emb, neg_cand_emb)
    pair = pair_metrics(pos_scores, neg_scores)

    pos_target_ids = [r["matchContactId"] for r in positives]
    retrieval = retrieval_metrics(
        query_embs=pos_seeker_emb,
        target_ids=pos_target_ids,
        candidate_ids=corpus_ids,
        candidate_embs=corpus_emb,
    )
    slices = slice_metrics(
        positives=positives,
        negatives=negatives,
        pos_scores=pos_scores,
        neg_scores=neg_scores,
        neg_seeker_texts=neg_seeker_texts,
        neg_cand_texts=neg_cand_texts,
        query_embs=pos_seeker_emb,
        target_ids=pos_target_ids,
        candidate_ids=corpus_ids,
        candidate_embs=corpus_emb,
    )
    print(
        f"    pair AUC {pair['roc_auc']:.4f} | "
        f"MRR {retrieval['mrr']:.4f} | R@1 {retrieval['recall@1']:.4f}"
    )
    return {
        "subset": subset,
        "n_candidates": len(corpus_ids),
        "real_data_hash": combined_hash,
        "pair": pair,
        "retrieval": retrieval,
        "slices": slices,
    }


def _run_tfidf_subsets(
    *,
    label: str,
    data_dir: Path,
    split_path: Path,
    subsets: Sequence[Subset],
    batch_size: int,
    cache_dir: Path,
    max_features: int,
    ngram_range: tuple[int, int],
) -> dict[Subset, dict[str, Any]]:
    """TF-IDF's vocabulary/IDF is corpus-dependent, so unlike every other kind it
    must be refit on each subset's own corpus — there is no shared encoding to
    reuse across subsets here. One full load + fit + encode per subset, exactly
    as ``baselines/tfidf/eval.py --holdout-only`` does, which is what
    ``tests/test_eval_real_full_baselines.py`` reproduces digit-for-digit.
    """
    out: dict[Subset, dict[str, Any]] = {}
    for subset in subsets:
        ps = load_real_pairs(data_dir, split_path, subset=subset, verify=True)
        positives, negatives = split_by_label(ps.pairs)
        print(
            f"\n=== {label} | subset={subset} | n={len(ps.pairs)} "
            f"(pos={len(positives)}, neg={len(negatives)}) ==="
        )

        pos_seeker_texts = [
            seeker_to_text(r["userContactFile"], r["searchQuery"]) for r in positives
        ]
        neg_seeker_texts = [
            seeker_to_text(r["userContactFile"], r["searchQuery"]) for r in negatives
        ]
        pos_cand_texts = [candidate_to_text(r["matchContactFile"]) for r in positives]
        neg_cand_texts = [candidate_to_text(r["matchContactFile"]) for r in negatives]

        corpus_ids, corpus_texts = build_candidate_corpus(positives, negatives)
        print(f"candidate corpus size: {len(corpus_ids)}")

        encoder = _build_encoder(
            "tfidf",
            model_name=None,
            device="cpu",
            max_length=0,
            truncate_dim=None,
            dtype="auto",
            cache_dir=cache_dir,
            max_features=max_features,
            ngram_range=ngram_range,
            fit_texts=pos_seeker_texts + neg_seeker_texts + corpus_texts,
        )
        adapter = _EncoderAdapter("tfidf", encoder, batch_size)

        out[subset] = _score(
            label=label,
            subset=subset,
            positives=positives,
            negatives=negatives,
            pos_seeker_emb=adapter.encode(pos_seeker_texts, "query"),
            neg_seeker_emb=adapter.encode(neg_seeker_texts, "query"),
            pos_cand_emb=adapter.encode(pos_cand_texts, "document"),
            neg_cand_emb=adapter.encode(neg_cand_texts, "document"),
            corpus_ids=corpus_ids,
            corpus_emb=adapter.encode(corpus_texts, "document"),
            neg_seeker_texts=neg_seeker_texts,
            neg_cand_texts=neg_cand_texts,
            combined_hash=ps.combined_hash,
        )
    return out


def _run_neural_subsets(
    *,
    kind: Kind,
    label: str,
    data_dir: Path,
    split_path: Path,
    subsets: Sequence[Subset],
    model_name: str | None,
    batch_size: int,
    max_length: int,
    truncate_dim: int | None,
    dtype: str,
    device: str,
    cache_dir: Path,
) -> dict[Subset, dict[str, Any]]:
    """Frozen bert/hf encoders: no fitting step, so seeker text depends only on
    its pair — every subset is a row mask over ``all``, never a different
    encode. One model load, one encode pass over the full 200 real pairs,
    reused for every requested subset.
    """
    ps_all = load_real_pairs(data_dir, split_path, subset="all", verify=True)
    ordered = list(ps_all.pairs)  # load_real_pairs already sorts by pair_id
    pos_ordered = [p for p in ordered if p.label == "pos"]
    neg_ordered = [p for p in ordered if p.label == "neg"]

    pos_seeker_texts_all = [
        seeker_to_text(p.pair["userContactFile"], p.pair["searchQuery"]) for p in pos_ordered
    ]
    neg_seeker_texts_all = [
        seeker_to_text(p.pair["userContactFile"], p.pair["searchQuery"]) for p in neg_ordered
    ]
    pos_cand_texts_all = [candidate_to_text(p.pair["matchContactFile"]) for p in pos_ordered]
    neg_cand_texts_all = [candidate_to_text(p.pair["matchContactFile"]) for p in neg_ordered]

    encoder = _build_encoder(
        kind,
        model_name=model_name,
        device=device,
        max_length=max_length,
        truncate_dim=truncate_dim,
        dtype=dtype,
        cache_dir=cache_dir,
        max_features=0,
        ngram_range=(1, 1),
        fit_texts=(),  # unused by bert/hf — frozen encoders, no fitting step
    )
    adapter = _EncoderAdapter(kind, encoder, batch_size)

    pos_seeker_emb_all = adapter.encode(pos_seeker_texts_all, "query")
    neg_seeker_emb_all = adapter.encode(neg_seeker_texts_all, "query")

    # Candidate texts, deduped by exact string across both pos and neg. Every
    # text any subset's build_candidate_corpus could ever first-see is in this
    # union (subsets are pair-id subsets of "all"), so lookup by string can
    # never miss regardless of which occurrence a given subset picks as first.
    doc_texts: list[str] = []
    seen: set[str] = set()
    for t in pos_cand_texts_all + neg_cand_texts_all:
        if t not in seen:
            seen.add(t)
            doc_texts.append(t)
    doc_matrix = adapter.encode(doc_texts, "document")
    doc_by_text = {t: doc_matrix[i] for i, t in enumerate(doc_texts)}

    out: dict[Subset, dict[str, Any]] = {}
    for subset in subsets:
        if subset == "all":
            pos_sel = list(enumerate(pos_ordered))
            neg_sel = list(enumerate(neg_ordered))
        else:
            wanted_source = _SOURCE_BY_SUBSET[subset]
            pos_sel = [(i, p) for i, p in enumerate(pos_ordered) if p.source == wanted_source]
            neg_sel = [(i, p) for i, p in enumerate(neg_ordered) if p.source == wanted_source]
        pos_rows = [i for i, _ in pos_sel]
        neg_rows = [i for i, _ in neg_sel]
        positives = [p.pair for _, p in pos_sel]
        negatives = [p.pair for _, p in neg_sel]
        print(
            f"\n=== {label} | subset={subset} | n={len(positives) + len(negatives)} "
            f"(pos={len(positives)}, neg={len(negatives)}) ==="
        )

        corpus_ids, corpus_texts = build_candidate_corpus(positives, negatives)
        print(f"candidate corpus size: {len(corpus_ids)}")

        neg_seeker_texts = [neg_seeker_texts_all[i] for i in neg_rows]
        neg_cand_texts = [neg_cand_texts_all[i] for i in neg_rows]

        out[subset] = _score(
            label=label,
            subset=subset,
            positives=positives,
            negatives=negatives,
            pos_seeker_emb=pos_seeker_emb_all[pos_rows],
            neg_seeker_emb=neg_seeker_emb_all[neg_rows],
            pos_cand_emb=np.stack([doc_by_text[pos_cand_texts_all[i]] for i in pos_rows]),
            neg_cand_emb=np.stack([doc_by_text[neg_cand_texts_all[i]] for i in neg_rows]),
            corpus_ids=corpus_ids,
            corpus_emb=np.stack([doc_by_text[t] for t in corpus_texts]),
            neg_seeker_texts=neg_seeker_texts,
            neg_cand_texts=neg_cand_texts,
            combined_hash=ps_all.combined_hash,
        )
    return out


def run_baseline_eval(
    *,
    kind: Kind,
    data_dir: Path,
    split_path: Path,
    label: str,
    model_name: str | None = None,
    subsets: Sequence[Subset] = DEFAULT_SUBSETS,
    batch_size: int = 8,
    max_length: int = 8192,
    truncate_dim: int | None = None,
    dtype: str = "auto",
    device: str = "cpu",
    cache_dir: Path = Path("/tmp/eval_real_full_cache"),
    max_features: int = 20000,
    ngram_range: tuple[int, int] = (1, 2),
) -> dict[str, Any]:
    """Evaluate one frozen baseline over each requested subset of the real pairs."""
    if kind == "tfidf":
        subset_metrics = _run_tfidf_subsets(
            label=label,
            data_dir=data_dir,
            split_path=split_path,
            subsets=subsets,
            batch_size=batch_size,
            cache_dir=cache_dir,
            max_features=max_features,
            ngram_range=ngram_range,
        )
    else:
        subset_metrics = _run_neural_subsets(
            kind=kind,
            label=label,
            data_dir=data_dir,
            split_path=split_path,
            subsets=subsets,
            model_name=model_name,
            batch_size=batch_size,
            max_length=max_length,
            truncate_dim=truncate_dim,
            dtype=dtype,
            device=device,
            cache_dir=cache_dir,
        )

    return {
        "label": label,
        "kind": kind,
        "model_name": model_name,
        "device": device,
        "batch_size": batch_size,
        "max_length": max_length,
        "truncate_dim": truncate_dim,
        "dtype": dtype,
        "subsets": subset_metrics,
    }
