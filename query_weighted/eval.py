"""Score every seeker-representation arm on the real pairs.

Encoding strategy — why everything is encoded once on the full 200
------------------------------------------------------------------
Seeker text depends only on its pair, so the ``train``/``holdout`` subsets are
row masks over the ``all`` ordering, not separate encodes. Candidate text is
identical across every arm, so the document side is encoded exactly once for the
whole experiment and reused by all of them.

That leaves the α-weighting family costing nothing at all: encode ``profile_only``
and ``query_only`` once each, and every α is ``normalize(α·Q̂ + (1−α)·P̂)`` in
numpy. Only the text-level arms (front-loading, repetition) need their own
encodes.

Candidate vectors are looked up by exact text string rather than by row index,
so a subset whose first-seen text for a contact id differs from the full set's
can never be silently mismatched.

One invariant worth stating loudly
----------------------------------
``baselines.metrics.slice_metrics`` uses ``neg_seeker_texts`` *only* to define
the easy/hard negative split by Jaccard token overlap. If each arm passed its own
seeker text, repeating the query three times would change the overlap and
therefore which negatives count as "hard" — every arm would be scored on a
different population and hard-neg AUC would be meaningless across the table. So
the hardness split is pinned to the ``concat_baseline`` text for every arm.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from baselines.metrics import pair_metrics, retrieval_metrics, slice_metrics
from baselines.voyage_nano.encode import cosine_scores, l2_normalize
from eval_real_full.baseline_eval import build_candidate_corpus, split_by_label
from eval_real_full.data import Subset, load_real_pairs

from query_weighted import text as qtext

DEFAULT_SUBSETS: tuple[Subset, ...] = ("all", "train", "holdout")

# Seeker builders that need their own encode pass. Each takes
# (userContactFile, searchQuery) and returns the string to embed.
TEXT_ARMS: dict[str, Callable[[Any, str], str]] = {
    "profile_only": qtext.profile_only,
    "concat_baseline": qtext.concat_baseline,
    "query_only": qtext.query_only,
    "query_first": qtext.query_first,
    "query_x3_front": lambda p, q: qtext.query_repeated_front(p, q, repeats=3),
    "query_x5_front": lambda p, q: qtext.query_repeated_front(p, q, repeats=5),
    "query_x10_front": lambda p, q: qtext.query_repeated_front(p, q, repeats=10),
}

# α for normalize(α·query_vec + (1−α)·profile_vec). 0.0 and 1.0 are omitted:
# they reproduce profile_only and query_only exactly, which are already arms.
ALPHAS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


@dataclass(frozen=True)
class Encoded:
    """Seeker matrices per arm plus the shared candidate lookup, on the full 200."""

    pair_ids: list[str]
    labels: list[str]
    positives: list[dict]
    negatives: list[dict]
    pos_index: list[int]
    neg_index: list[int]
    seeker: dict[str, np.ndarray]  # arm -> (200, d), rows aligned to pair_ids
    doc_by_text: dict[str, np.ndarray]


def _seeker_texts(pairs: list[dict], builder: Callable[[Any, str], str]) -> list[str]:
    return [builder(p["userContactFile"], p["searchQuery"]) for p in pairs]


def encode_everything(
    encoder,
    data_dir: Path,
    split_path: Path,
    *,
    batch_size: int = 4,
    arms: Sequence[str] | None = None,
) -> Encoded:
    """One pass over the full 200 pairs producing every vector the arms need."""
    ps = load_real_pairs(data_dir, split_path, subset="all", verify=True)
    ordered = list(ps.pairs)
    pair_ids = [p.pair_id for p in ordered]
    labels = [p.label for p in ordered]
    raw = [p.pair for p in ordered]

    positives, negatives = split_by_label(ordered)
    pos_index = [i for i, lab in enumerate(labels) if lab == "pos"]
    neg_index = [i for i, lab in enumerate(labels) if lab == "neg"]

    wanted = list(arms) if arms is not None else list(TEXT_ARMS)
    seeker: dict[str, np.ndarray] = {}
    for arm in wanted:
        texts = _seeker_texts(raw, TEXT_ARMS[arm])
        print(f"  encoding seeker arm {arm!r} ({len(texts)} texts)")
        seeker[arm] = encoder.encode(
            texts, role="query", batch_size=batch_size, show_progress=False
        )

    # Candidate side: identical for every arm. Encode the distinct texts once and
    # look them up by string, so ordering differences between subsets cannot
    # cause a mismatch.
    doc_texts: list[str] = []
    seen: set[str] = set()
    for record in raw:
        t = qtext.candidate_to_text(record["matchContactFile"])
        if t not in seen:
            seen.add(t)
            doc_texts.append(t)
    print(f"  encoding {len(doc_texts)} distinct candidate texts (shared by all arms)")
    doc_matrix = encoder.encode(
        doc_texts, role="document", batch_size=batch_size, show_progress=False
    )
    doc_by_text = {t: doc_matrix[i] for i, t in enumerate(doc_texts)}

    return Encoded(
        pair_ids=pair_ids,
        labels=labels,
        positives=positives,
        negatives=negatives,
        pos_index=pos_index,
        neg_index=neg_index,
        seeker=seeker,
        doc_by_text=doc_by_text,
    )


def combine(query_vecs: np.ndarray, profile_vecs: np.ndarray, alpha: float) -> np.ndarray:
    """normalize(α·Q̂ + (1−α)·P̂) — both inputs are already unit vectors."""
    return l2_normalize(alpha * query_vecs + (1.0 - alpha) * profile_vecs).astype(np.float32)


def _docs_for(records: list[dict], doc_by_text: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack([doc_by_text[qtext.candidate_to_text(r["matchContactFile"])] for r in records])


def score_arm(
    enc: Encoded,
    seeker_all: np.ndarray,
    subset_pair_ids: set[str],
) -> dict[str, Any]:
    """Metrics for one seeker representation restricted to one subset."""
    # split_by_label preserves the full-set ordering, so positives[k] is the
    # record at row pos_index[k]; keep the two aligned when filtering.
    pos_sel = [(k, i) for k, i in enumerate(enc.pos_index) if enc.pair_ids[i] in subset_pair_ids]
    neg_sel = [(k, i) for k, i in enumerate(enc.neg_index) if enc.pair_ids[i] in subset_pair_ids]
    positives = [enc.positives[k] for k, _ in pos_sel]
    negatives = [enc.negatives[k] for k, _ in neg_sel]
    pos_rows = [i for _, i in pos_sel]
    neg_rows = [i for _, i in neg_sel]

    pos_seeker = seeker_all[pos_rows]
    neg_seeker = seeker_all[neg_rows]
    pos_cand = _docs_for(positives, enc.doc_by_text)
    neg_cand = _docs_for(negatives, enc.doc_by_text)

    corpus_ids, corpus_texts = build_candidate_corpus(positives, negatives)
    corpus_embs = np.stack([enc.doc_by_text[t] for t in corpus_texts])

    pos_scores = cosine_scores(pos_seeker, pos_cand)
    neg_scores = cosine_scores(neg_seeker, neg_cand)
    target_ids = [r["matchContactId"] for r in positives]

    # Hardness split pinned to the baseline text — see module docstring.
    hardness_neg_seeker = _seeker_texts(negatives, qtext.concat_baseline)
    hardness_neg_cand = [qtext.candidate_to_text(r["matchContactFile"]) for r in negatives]

    return {
        "n_candidates": len(corpus_ids),
        "n_pos": len(positives),
        "n_neg": len(negatives),
        "pair": pair_metrics(pos_scores, neg_scores),
        "retrieval": retrieval_metrics(
            query_embs=pos_seeker,
            target_ids=target_ids,
            candidate_ids=corpus_ids,
            candidate_embs=corpus_embs,
        ),
        "slices": slice_metrics(
            positives=positives,
            negatives=negatives,
            pos_scores=pos_scores,
            neg_scores=neg_scores,
            neg_seeker_texts=hardness_neg_seeker,
            neg_cand_texts=hardness_neg_cand,
            query_embs=pos_seeker,
            target_ids=target_ids,
            candidate_ids=corpus_ids,
            candidate_embs=corpus_embs,
        ),
    }


def run_all_arms(
    encoder,
    data_dir: Path,
    split_path: Path,
    *,
    subsets: Sequence[Subset] = DEFAULT_SUBSETS,
    batch_size: int = 4,
    alphas: Sequence[float] = ALPHAS,
) -> dict[str, Any]:
    enc = encode_everything(
        encoder, data_dir, split_path, batch_size=batch_size
    )

    representations: dict[str, np.ndarray] = dict(enc.seeker)
    for alpha in alphas:
        representations[f"alpha_{alpha:.1f}"] = combine(
            enc.seeker["query_only"], enc.seeker["profile_only"], alpha
        )

    subset_ids: dict[str, set[str]] = {}
    for subset in subsets:
        ps = load_real_pairs(data_dir, split_path, subset=subset, verify=True)
        subset_ids[subset] = {p.pair_id for p in ps.pairs}

    out: dict[str, Any] = {
        "model_name": getattr(encoder, "model_name", "unknown"),
        "max_length": getattr(encoder, "max_length", None),
        "truncate_dim": getattr(encoder, "truncate_dim", None),
        "alphas": list(alphas),
        "hardness_split": "pinned to concat_baseline seeker text for every arm",
        "arms": {},
    }
    for name, seeker_all in representations.items():
        out["arms"][name] = {}
        for subset in subsets:
            m = score_arm(enc, seeker_all, subset_ids[subset])
            out["arms"][name][subset] = m
        a = out["arms"][name]["all"]
        print(
            f"{name:18s} all-200  AUC={a['pair']['roc_auc']:.4f} "
            f"MRR={a['retrieval']['mrr']:.4f} R@1={a['retrieval']['recall@1']:.4f} "
            f"R@10={a['retrieval']['recall@10']:.4f}"
        )
    return out


def write_metrics(metrics: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "metrics.json"
    path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nwrote {path}")
    return path
