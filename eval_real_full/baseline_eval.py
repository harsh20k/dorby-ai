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

import json
from pathlib import Path
from typing import Any, Literal, Sequence

from baselines.bert_frozen.text import candidate_to_text, seeker_to_text
from baselines.metrics import pair_metrics, retrieval_metrics, slice_metrics

from eval_real_full.data import Subset, load_real_pairs

Kind = Literal["tfidf", "bert", "hf"]

DEFAULT_SUBSETS: tuple[Subset, ...] = ("all", "train", "holdout")


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
    out: dict[str, Any] = {
        "label": label,
        "kind": kind,
        "model_name": model_name,
        "device": device,
        "batch_size": batch_size,
        "max_length": max_length,
        "truncate_dim": truncate_dim,
        "dtype": dtype,
        "subsets": {},
    }

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

        # Rebuilt per subset: TF-IDF must refit, and a neural encoder's cache
        # directory is keyed by content anyway, so reloading costs only the
        # model load — which the HF cache volume makes cheap.
        encoder = _build_encoder(
            kind,
            model_name=model_name,
            device=device,
            max_length=max_length,
            truncate_dim=truncate_dim,
            dtype=dtype,
            cache_dir=cache_dir,
            max_features=max_features,
            ngram_range=ngram_range,
            fit_texts=pos_seeker_texts + neg_seeker_texts + corpus_texts,
        )
        adapter = _EncoderAdapter(kind, encoder, batch_size)

        pos_seeker_emb = adapter.encode(pos_seeker_texts, "query")
        neg_seeker_emb = adapter.encode(neg_seeker_texts, "query")
        pos_cand_emb = adapter.encode(pos_cand_texts, "document")
        neg_cand_emb = adapter.encode(neg_cand_texts, "document")
        corpus_emb = adapter.encode(corpus_texts, "document")

        # Every encoder here returns L2-normalized rows, so the shared
        # dot-product cosine applies — same as each baseline's own eval.py.
        from baselines.hf_embedding.encode import cosine_scores

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

        metrics = {
            "subset": subset,
            "n_candidates": len(corpus_ids),
            "real_data_hash": ps.combined_hash,
            "pair": pair,
            "retrieval": retrieval,
            "slices": slices,
        }
        out["subsets"][subset] = metrics
        print(
            f"    pair AUC {pair['roc_auc']:.4f} | "
            f"MRR {retrieval['mrr']:.4f} | R@1 {retrieval['recall@1']:.4f}"
        )
    return out


def write_metrics(metrics: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "metrics.json"
    path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nwrote {path}")
    return path
