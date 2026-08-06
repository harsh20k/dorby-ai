"""Score the three identity-field-pair arms on the real pairs.

Structurally adapted from ``query_weighted/eval.py`` (own ``TEXT_ARMS``, own
orchestration — the two-field builders aren't a variant of anything that
file already does) but every low-level scoring primitive is imported and
called unmodified: ``baselines.metrics``, ``baselines.voyage_nano.encode``,
``eval_real_full``'s 200-pair loader. Candidate text is unchanged from every
other experiment, so it hits the shared embedding cache for free when one is
mounted.

Hardness split: pinned to ``baselines.bert_frozen.text.seeker_to_text`` (the
full profile + query, exactly what every published baseline uses to define
easy/hard negatives) so this experiment's hard-neg AUC is comparable to every
other row in the project, not to a hardness split defined by its own
two-field text.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from baselines.bert_frozen.text import seeker_to_text
from baselines.metrics import pair_metrics, retrieval_metrics, slice_metrics
from baselines.voyage_nano.encode import cosine_scores
from eval_real_full.baseline_eval import build_candidate_corpus, split_by_label
from eval_real_full.data import Subset, load_real_pairs

from field_pairs_sweep import text as fptext

DEFAULT_SUBSETS: tuple[Subset, ...] = ("all", "train", "holdout")

TEXT_ARMS: dict[str, Callable[[Any, str], str]] = {
    "pos_background": fptext.pos_background,
    "pos_lookingfor": fptext.pos_lookingfor,
    "background_lookingfor": fptext.background_lookingfor,
}


@dataclass(frozen=True)
class Encoded:
    pair_ids: list[str]
    labels: list[str]
    positives: list[dict]
    negatives: list[dict]
    pos_index: list[int]
    neg_index: list[int]
    seeker: dict[str, np.ndarray]
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

    doc_texts: list[str] = []
    seen: set[str] = set()
    for record in raw:
        t = fptext.candidate_to_text(record["matchContactFile"])
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


def _docs_for(records: list[dict], doc_by_text: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack([doc_by_text[fptext.candidate_to_text(r["matchContactFile"])] for r in records])


def score_arm(
    enc: Encoded,
    seeker_all: np.ndarray,
    subset_pair_ids: set[str],
) -> dict[str, Any]:
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

    # Hardness split pinned to the full-profile+query baseline text — see
    # module docstring.
    hardness_neg_seeker = [
        seeker_to_text(r["userContactFile"], r["searchQuery"]) for r in negatives
    ]
    hardness_neg_cand = [fptext.candidate_to_text(r["matchContactFile"]) for r in negatives]

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
) -> dict[str, Any]:
    enc = encode_everything(encoder, data_dir, split_path, batch_size=batch_size)

    subset_ids: dict[str, set[str]] = {}
    for subset in subsets:
        ps = load_real_pairs(data_dir, split_path, subset=subset, verify=True)
        subset_ids[subset] = {p.pair_id for p in ps.pairs}

    out: dict[str, Any] = {
        "model_name": getattr(encoder, "model_name", "unknown"),
        "max_length": getattr(encoder, "max_length", None),
        "truncate_dim": getattr(encoder, "truncate_dim", None),
        "hardness_split": "pinned to full profile+query seeker text for every arm",
        "arms": {},
    }
    for name, seeker_all in enc.seeker.items():
        out["arms"][name] = {}
        for subset in subsets:
            m = score_arm(enc, seeker_all, subset_ids[subset])
            out["arms"][name][subset] = m
        a = out["arms"][name]["all"]
        print(
            f"{name:22s} all-200  AUC={a['pair']['roc_auc']:.4f} "
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
