"""Offline eval for an open-weight HF embedding model, field-selected text
(matches the focused LLM judge). Isolated variation of
baselines/hf_embedding/eval.py — same encoder machinery, different text
packing (see text.py), and restricted to the 200 real seed pairs (not the
full canonical dataset, which also holds 460 promoted synthetic pairs).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from baselines.hf_embedding.encode import cosine_scores, get_encoder_class, pick_device
from baselines.hf_embedding.models import slugify
from baselines.hf_embedding_field_selected.text import candidate_to_text, seeker_to_text
from baselines.llm_judge.real_pairs import load_real_pairs
from baselines.metrics import pair_metrics, print_metrics, retrieval_metrics, slice_metrics


def build_candidate_corpus(
    positives: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
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


def run_eval(
    data_dir: Path,
    model_name: str,
    batch_size: int,
    max_length: int,
    truncate_dim: int | None,
    dtype: str,
    artifacts_dir: Path,
    *,
    holdout_only: bool = False,
    split_path: Path | None = None,
) -> dict[str, Any]:
    device = pick_device()
    print(f"device: {device}")
    print(f"model:  {model_name}")
    print("text:   field-selected (seeker: positioning+lookingFor+query; "
          "candidate: positioning+background+lookingFor)")

    split = "holdout" if holdout_only else "all"
    positives, negatives = load_real_pairs(data_dir, split=split, split_path=split_path)
    print(f"loaded {len(positives)} real positives, {len(negatives)} real negatives (split={split})")

    pos_seeker_texts = [seeker_to_text(r["userContactFile"], r["searchQuery"]) for r in positives]
    neg_seeker_texts = [seeker_to_text(r["userContactFile"], r["searchQuery"]) for r in negatives]
    pos_cand_texts = [candidate_to_text(r["matchContactFile"]) for r in positives]
    neg_cand_texts = [candidate_to_text(r["matchContactFile"]) for r in negatives]

    corpus_ids, corpus_texts = build_candidate_corpus(positives, negatives)
    print(f"candidate corpus size: {len(corpus_ids)}")

    EncoderClass = get_encoder_class(model_name)
    encoder = EncoderClass(
        model_name=model_name,
        device=device,
        max_length=max_length,
        truncate_dim=truncate_dim,
        dtype=dtype,
        cache_dir=artifacts_dir,
    )

    pos_seeker_emb = encoder.encode(
        pos_seeker_texts, role="query", batch_size=batch_size, cache_name="pos_seeker"
    )
    neg_seeker_emb = encoder.encode(
        neg_seeker_texts, role="query", batch_size=batch_size, cache_name="neg_seeker"
    )
    pos_cand_emb = encoder.encode(
        pos_cand_texts, role="document", batch_size=batch_size, cache_name="pos_cand"
    )
    neg_cand_emb = encoder.encode(
        neg_cand_texts, role="document", batch_size=batch_size, cache_name="neg_cand"
    )
    corpus_emb = encoder.encode(
        corpus_texts, role="document", batch_size=batch_size, cache_name="corpus"
    )

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

    return {
        "model_name": model_name,
        "device": str(device),
        "max_length": max_length,
        "truncate_dim": encoder.truncate_dim,
        "dtype": dtype,
        "batch_size": batch_size,
        "seeker_text": "field_selected",
        "seeker_fields": ["positioning", "lookingFor"],
        "candidate_fields": ["positioning", "background", "lookingFor"],
        "holdout_only": holdout_only,
        "pair": pair,
        "retrieval": retrieval,
        "slices": slices,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--model", type=str, required=True, help="HF model id, e.g. Qwen/Qwen3-Embedding-8B")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--truncate-dim", type=int, default=None)
    p.add_argument("--dtype", type=str, default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    p.add_argument("--artifacts-dir", type=Path, default=None)
    p.add_argument("--holdout-only", action="store_true")
    p.add_argument("--split-path", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    artifacts_dir = args.artifacts_dir or Path("artifacts") / f"hf_embedding_field_selected_{slugify(args.model)}"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    metrics = run_eval(
        data_dir=args.data_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        truncate_dim=args.truncate_dim,
        dtype=args.dtype,
        artifacts_dir=artifacts_dir,
        holdout_only=args.holdout_only,
        split_path=args.split_path,
    )
    print_metrics(metrics)

    out_path = artifacts_dir / "metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
