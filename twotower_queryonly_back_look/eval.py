"""Score the adapter on all 200 real pairs — seeker = query only, candidate
= background+lookingFor, matching what the model was trained on.

Same reason as `twotower_field_pos_bg/eval.py`: the real-pairs loader
(`eval_real_full.eval.run_eval`) always builds full-profile text on both
sides, so it can't score this adapter fairly. Model loading and
query/document encoding (`load_model_for_eval`, `encode_role`) are reused
unmodified from `twotower.eval`; only text extraction and the candidate
corpus are specific to this experiment — and even those are built from two
already-existing, unmodified builders (`query_weighted.text.query_only`,
`field_pairs_sweep.text.background_lookingfor`), not new code.

Hardness split is pinned to the full profile+query baseline text
(`baselines.bert_frozen.text`), same convention as every other experiment
in this family, so hard-neg AUC stays comparable project-wide even though
the scored vectors are trimmed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from baselines.bert_frozen.text import candidate_to_text as full_candidate_to_text
from baselines.bert_frozen.text import seeker_to_text as full_seeker_to_text
from baselines.metrics import pair_metrics, retrieval_metrics, slice_metrics
from baselines.voyage_nano.encode import cosine_scores
from eval_real_full.baseline_eval import split_by_label
from eval_real_full.data import Subset, load_real_pairs
from field_pairs_sweep.text import background_lookingfor
from query_weighted.text import query_only
from twotower.eval import encode_role, load_model_for_eval

DEFAULT_SUBSETS: tuple[Subset, ...] = ("all", "train", "holdout")


def _trimmed_corpus(positives: list[dict], negatives: list[dict]) -> tuple[list[str], list[str]]:
    ids: list[str] = []
    texts: list[str] = []
    seen: set[str] = set()
    for record in positives + negatives:
        mid = record["matchContactId"]
        if mid in seen:
            continue
        seen.add(mid)
        ids.append(mid)
        texts.append(background_lookingfor(record["matchContactFile"]))
    return ids, texts


def evaluate_subset(model, positives: list[dict], negatives: list[dict], *, batch_size: int = 8) -> dict[str, Any]:
    pos_seeker = [query_only(r["userContactFile"], r["searchQuery"]) for r in positives]
    neg_seeker = [query_only(r["userContactFile"], r["searchQuery"]) for r in negatives]
    pos_cand = [background_lookingfor(r["matchContactFile"]) for r in positives]
    neg_cand = [background_lookingfor(r["matchContactFile"]) for r in negatives]
    corpus_ids, corpus_texts = _trimmed_corpus(positives, negatives)

    pos_seeker_emb = encode_role(model, pos_seeker, role="query", batch_size=batch_size)
    neg_seeker_emb = encode_role(model, neg_seeker, role="query", batch_size=batch_size)
    pos_cand_emb = encode_role(model, pos_cand, role="document", batch_size=batch_size)
    neg_cand_emb = encode_role(model, neg_cand, role="document", batch_size=batch_size)
    corpus_emb = encode_role(model, corpus_texts, role="document", batch_size=batch_size)

    pos_scores = cosine_scores(pos_seeker_emb, pos_cand_emb)
    neg_scores = cosine_scores(neg_seeker_emb, neg_cand_emb)
    pos_target_ids = [r["matchContactId"] for r in positives]

    hardness_neg_seeker = [full_seeker_to_text(r["userContactFile"], r["searchQuery"]) for r in negatives]
    hardness_neg_cand = [full_candidate_to_text(r["matchContactFile"]) for r in negatives]

    return {
        "n_pos": len(positives),
        "n_neg": len(negatives),
        "n_candidates": len(corpus_ids),
        "pair": pair_metrics(pos_scores, neg_scores),
        "retrieval": retrieval_metrics(
            query_embs=pos_seeker_emb,
            target_ids=pos_target_ids,
            candidate_ids=corpus_ids,
            candidate_embs=corpus_emb,
        ),
        "slices": slice_metrics(
            positives=positives,
            negatives=negatives,
            pos_scores=pos_scores,
            neg_scores=neg_scores,
            neg_seeker_texts=hardness_neg_seeker,
            neg_cand_texts=hardness_neg_cand,
            query_embs=pos_seeker_emb,
            target_ids=pos_target_ids,
            candidate_ids=corpus_ids,
            candidate_embs=corpus_emb,
        ),
    }


def run_eval(
    *,
    data_dir: Path,
    split_path: Path,
    model_name: str,
    adapter_dir: Path,
    batch_size: int = 8,
    device: str = "cuda",
    truncate_dim: int = 1024,
    max_length: int = 4096,
    subsets: tuple[Subset, ...] = DEFAULT_SUBSETS,
) -> dict[str, Any]:
    model = load_model_for_eval(
        model_name=model_name,
        adapter_dir=adapter_dir,
        device=device,
        max_seq_length=max_length,
        truncate_dim=truncate_dim,
    )
    out: dict[str, Any] = {
        "model_name": model_name,
        "adapter_dir": str(adapter_dir),
        "seeker_fields": [],
        "seeker_uses_query": True,
        "candidate_fields": ["background", "lookingFor"],
        "subsets": {},
    }
    for subset in subsets:
        ps = load_real_pairs(data_dir, split_path, subset=subset, verify=True)
        positives, negatives = split_by_label(ps.pairs)
        m = evaluate_subset(model, positives, negatives, batch_size=batch_size)
        out["subsets"][subset] = m
        print(
            f"{subset:8s} AUC={m['pair']['roc_auc']:.4f} "
            f"hard={m['slices']['neg_hardness']['hard']['pair_auc']:.4f} "
            f"MRR={m['retrieval']['mrr']:.4f} R@1={m['retrieval']['recall@1']:.4f}"
        )
    return out


def write_metrics(metrics: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "metrics.json"
    path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"wrote {path}")
    return path
