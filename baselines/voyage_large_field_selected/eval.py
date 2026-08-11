"""Offline eval for Voyage-4-large API, field-selected text (matches the focused LLM judge).

Isolated variation of baselines/voyage_large/eval.py: same API encoder
(``VoyageLargeEncoder``), same metrics, only the text packing changes (see
text.py). Defaults to the full 200-pair real dataset, not holdout-only —
pass --holdout-only for the matched-population comparison instead.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from baselines.llm_judge.real_pairs import load_real_pairs
from baselines.metrics import pair_metrics, print_metrics, retrieval_metrics, slice_metrics
from baselines.voyage_large.encode import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_RPM_LIMIT,
    DEFAULT_TPM_LIMIT,
    VoyageLargeEncoder,
    cosine_scores,
    estimate_tokens,
    require_api_key,
)
from baselines.voyage_large_field_selected.text import candidate_to_text, seeker_to_text


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


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"warning: invalid {name}={raw!r}, using default {default}")
        return default


def run_eval(
    data_dir: Path,
    model_name: str,
    output_dimension: int,
    batch_size: int,
    artifacts_dir: Path,
    tpm_limit: int,
    rpm_limit: int,
    *,
    holdout_only: bool = False,
    split_path: Path | None = None,
) -> dict[str, Any]:
    print(f"model:             {model_name}")
    print(f"output_dimension:  {output_dimension}")
    print(f"batch_size:        {batch_size}")
    print("text:              field-selected (seeker: positioning+lookingFor+query; "
          "candidate: positioning+background+lookingFor)")

    # load_real_pairs restricts to the 200 real seed pairs (train+holdout from
    # data/synthetic/seed_split.json), excluding the 460 cmsynth* promoted
    # synthetic pairs also present in dataset_positive/negative.json — unlike
    # baselines/voyage_large/eval.py's load_pairs, which loads everything.
    split = "holdout" if holdout_only else "all"
    positives, negatives = load_real_pairs(data_dir, split=split, split_path=split_path)
    print(f"loaded {len(positives)} real positives, {len(negatives)} real negatives (split={split})")

    pos_seeker_texts = [seeker_to_text(r["userContactFile"], r["searchQuery"]) for r in positives]
    neg_seeker_texts = [seeker_to_text(r["userContactFile"], r["searchQuery"]) for r in negatives]
    pos_cand_texts = [candidate_to_text(r["matchContactFile"]) for r in positives]
    neg_cand_texts = [candidate_to_text(r["matchContactFile"]) for r in negatives]

    corpus_ids, corpus_texts = build_candidate_corpus(positives, negatives)
    print(f"candidate corpus size: {len(corpus_ids)}")

    all_texts = pos_seeker_texts + neg_seeker_texts + pos_cand_texts + neg_cand_texts + corpus_texts
    est_tokens = estimate_tokens(all_texts)
    print(f"estimated tokens (naive, pre-dedupe/cache): ~{est_tokens:,}")

    encoder = VoyageLargeEncoder(
        model_name=model_name,
        output_dimension=output_dimension,
        truncation=True,
        cache_dir=artifacts_dir,
        tpm_limit=tpm_limit,
        rpm_limit=rpm_limit,
        batch_size=batch_size,
    )

    pos_seeker_emb = encoder.encode(
        pos_seeker_texts, input_type="query", batch_size=batch_size, label="pos_seeker"
    )
    neg_seeker_emb = encoder.encode(
        neg_seeker_texts, input_type="query", batch_size=batch_size, label="neg_seeker"
    )
    pos_cand_emb = encoder.encode(
        pos_cand_texts, input_type="document", batch_size=batch_size, label="pos_cand"
    )
    neg_cand_emb = encoder.encode(
        neg_cand_texts, input_type="document", batch_size=batch_size, label="neg_cand"
    )
    corpus_emb = encoder.encode(
        corpus_texts, input_type="document", batch_size=batch_size, label="corpus"
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

    usage = encoder.stats()
    usage_path = encoder.write_usage_meta()
    print(
        f"API usage: {usage['total_tokens_used']:,} tokens, "
        f"{usage['total_api_calls']} calls, "
        f"cache hits={usage['cache_hits']} misses={usage['cache_misses']}"
    )
    print(f"wrote {usage_path}")

    return {
        "model_name": model_name,
        "output_dimension": output_dimension,
        "batch_size": batch_size,
        "tpm_limit": tpm_limit,
        "rpm_limit": rpm_limit,
        "seeker_text": "field_selected",
        "seeker_fields": ["positioning", "lookingFor"],
        "candidate_fields": ["positioning", "background", "lookingFor"],
        "holdout_only": holdout_only,
        "pair": pair,
        "retrieval": retrieval,
        "slices": slices,
        "usage": usage,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--model", type=str, default="voyage-4-large")
    p.add_argument("--output-dimension", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--tpm-limit", type=int, default=_env_int("VOYAGE_TPM_LIMIT", DEFAULT_TPM_LIMIT))
    p.add_argument("--rpm-limit", type=int, default=_env_int("VOYAGE_RPM_LIMIT", DEFAULT_RPM_LIMIT))
    p.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/voyage_large_field_selected"))
    p.add_argument("--holdout-only", action="store_true")
    p.add_argument("--split-path", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    require_api_key()
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    metrics = run_eval(
        data_dir=args.data_dir,
        model_name=args.model,
        output_dimension=args.output_dimension,
        batch_size=args.batch_size,
        artifacts_dir=args.artifacts_dir,
        tpm_limit=args.tpm_limit,
        rpm_limit=args.rpm_limit,
        holdout_only=args.holdout_only,
        split_path=args.split_path,
    )
    print_metrics(metrics)

    out_path = args.artifacts_dir / "metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
