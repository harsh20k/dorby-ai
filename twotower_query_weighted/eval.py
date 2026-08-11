"""Score alpha-blended seeker encoding through the fine-tuned two-tower adapter.

Mirrors ``query_weighted/eval.py``'s design (encode profile-only and
query-only once, then every alpha is a free numpy blend) but swaps the frozen
HF encoder for ``twotower.eval.load_model_for_eval`` + ``encode_role``, so the
LoRA adapter is in the loop. Text builders (``profile_only`` / ``query_only`` /
``concat_baseline``) are imported unchanged from ``query_weighted.text`` — they
already wrap ``baselines.bert_frozen.text``, the same serialization
``twotower.data.LabeledPair.seeker_text`` uses, so ``concat_baseline`` output is
byte-identical to what the adapter was fine-tuned and originally evaluated on.

No file under ``twotower/``, ``query_weighted/``, or ``eval_real_full/`` is
modified — everything here is a new read of those modules' public API.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from baselines.metrics import pair_metrics, retrieval_metrics, slice_metrics
from baselines.voyage_nano.encode import cosine_scores, l2_normalize
from eval_real_full.data import Subset, load_real_pairs
from eval_real_full.guard import assert_trained_without_real_pairs
from twotower.eval import encode_role, load_model_for_eval

from query_weighted import text as qtext

DEFAULT_SUBSETS: tuple[Subset, ...] = ("all", "train", "holdout")

# Same sweep as query_weighted/eval.py — 0.0/1.0 omitted, they reproduce
# profile_only/query_only exactly.
ALPHAS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


@dataclass(frozen=True)
class Encoded:
    pair_ids: list[str]
    labels: list[str]
    positives: list[dict]
    negatives: list[dict]
    pos_index: list[int]
    neg_index: list[int]
    seeker: dict[str, np.ndarray]  # arm -> (n, d), rows aligned to pair_ids
    doc_by_text: dict[str, np.ndarray]


def load_adapter_model(
    *,
    model_name: str,
    adapter_dir: Path,
    device: str,
    max_length: int,
    truncate_dim: int,
) -> SentenceTransformer:
    """Loads the LoRA adapter exactly as ``eval_real_full/eval.py`` does.

    ``assert_trained_without_real_pairs`` fails loudly rather than silently
    producing a leaked, flattering number.
    """
    assert_trained_without_real_pairs(adapter_dir)
    return load_model_for_eval(
        model_name=model_name,
        adapter_dir=adapter_dir,
        device=device,
        max_seq_length=max_length,
        truncate_dim=truncate_dim,
    )


def encode_everything(
    model: SentenceTransformer,
    data_dir: Path,
    split_path: Path,
    *,
    batch_size: int = 8,
) -> Encoded:
    ps = load_real_pairs(data_dir, split_path, subset="all", verify=True)
    ordered = list(ps.pairs)
    pair_ids = [p.pair_id for p in ordered]
    labels = [p.label for p in ordered]
    raw = [p.pair for p in ordered]

    positives = [r for r, lab in zip(raw, labels) if lab == "pos"]
    negatives = [r for r, lab in zip(raw, labels) if lab == "neg"]
    pos_index = [i for i, lab in enumerate(labels) if lab == "pos"]
    neg_index = [i for i, lab in enumerate(labels) if lab == "neg"]

    seeker: dict[str, np.ndarray] = {}
    for arm, builder in (("profile_only", qtext.profile_only), ("query_only", qtext.query_only)):
        texts = [builder(p["userContactFile"], p["searchQuery"]) for p in raw]
        print(f"  encoding seeker arm {arm!r} ({len(texts)} texts)")
        seeker[arm] = encode_role(model, texts, role="query", batch_size=batch_size)

    doc_texts: list[str] = []
    seen: set[str] = set()
    for record in raw:
        t = qtext.candidate_to_text(record["matchContactFile"])
        if t not in seen:
            seen.add(t)
            doc_texts.append(t)
    print(f"  encoding {len(doc_texts)} distinct candidate texts")
    doc_matrix = encode_role(model, doc_texts, role="document", batch_size=batch_size)
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
    """normalize(alpha*query + (1-alpha)*profile) — both inputs unit vectors."""
    return l2_normalize(alpha * query_vecs + (1.0 - alpha) * profile_vecs).astype(np.float32)


def _docs_for(records: list[dict], doc_by_text: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack([doc_by_text[qtext.candidate_to_text(r["matchContactFile"])] for r in records])


def score_arm(enc: Encoded, seeker_all: np.ndarray, subset_pair_ids: set[str]) -> dict[str, Any]:
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

    corpus_ids: list[str] = []
    corpus_texts: list[str] = []
    seen: set[str] = set()
    for record in positives + negatives:
        mid = record["matchContactId"]
        if mid in seen:
            continue
        seen.add(mid)
        corpus_ids.append(mid)
        corpus_texts.append(qtext.candidate_to_text(record["matchContactFile"]))
    corpus_embs = np.stack([enc.doc_by_text[t] for t in corpus_texts])

    pos_scores = cosine_scores(pos_seeker, pos_cand)
    neg_scores = cosine_scores(neg_seeker, neg_cand)
    target_ids = [r["matchContactId"] for r in positives]

    # Hardness split pinned to the concat_baseline text, matching
    # query_weighted/eval.py, so every arm is scored on the same hard/easy
    # negative population.
    hardness_neg_seeker = [
        qtext.concat_baseline(r["userContactFile"], r["searchQuery"]) for r in negatives
    ]
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
    model: SentenceTransformer,
    data_dir: Path,
    split_path: Path,
    *,
    label: str,
    subsets: Sequence[Subset] = DEFAULT_SUBSETS,
    batch_size: int = 8,
    alphas: Sequence[float] = ALPHAS,
) -> dict[str, Any]:
    enc = encode_everything(model, data_dir, split_path, batch_size=batch_size)

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
        "label": label,
        "alphas": list(alphas),
        "hardness_split": "pinned to concat_baseline seeker text for every arm",
        "arms": {},
    }
    for name, seeker_all in representations.items():
        out["arms"][name] = {}
        for subset in subsets:
            out["arms"][name][subset] = score_arm(enc, seeker_all, subset_ids[subset])
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
