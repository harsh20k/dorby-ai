"""Static reciprocal two-tower baseline, lambda calibrated on rrf_003 judge labels.

Deliberate variant of ``baselines/reciprocal_static/eval.py`` (experiment
isolation rule — that experiment's numbers must stay reproducible from its
own code untouched, so this is a new package, not an edit). Two differences
from the original:

1. ``lambda`` is fit by the same 1-D grid search maximizing pair ROC-AUC, but
   on rrf_003's 2,619 judge-labeled synthetic pairs
   (``exports/rrf_datasets/rrf_003/``) instead of the 131 real train pairs.
   Those labels are a model's opinion (``google/gemini-3.1-flash-lite``,
   naive framing, 0.6358 holdout pair AUC as a labeler), not real
   accept/decline outcomes — see that batch's README before trusting this
   run's lambda at face value.
2. ``bg_text`` (the v_i / background view) narrows to ``positioning`` +
   ``background`` only, instead of every non-lookingFor profile field
   (baselines/reciprocal_static_rrf003/text.py).

The frozen lambda is then applied, unchanged, to the full 200 real
accept/decline pairs (data/dataset_positive.json + dataset_negative.json) —
real labels never touch fitting in this run, matching the train/holdout
discipline used everywhere else in this repo, just with the fitting
population swapped for a much larger synthetic one.

Usage:
  python -m baselines.reciprocal_static_rrf003.eval --data-dir data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from baselines.bert_frozen.text import candidate_to_text, seeker_to_text
from baselines.metrics import pair_metrics, retrieval_metrics, slice_metrics
from baselines.reciprocal_static.eval import build_lambda_grid, fit_lambda
from baselines.reciprocal_static_rrf003.rrf003_data import DEFAULT_RRF003_DIR, load_rrf003_pairs
from baselines.reciprocal_static_rrf003.text import bg_text, look_text, seeker_look_text
from baselines.voyage_nano.encode import VoyageNanoEncoder, cosine_scores, pick_device
from eval_real_full.data import load_real_pairs
from twotower.data import LabeledPair

DEFAULT_LAMBDA_MIN = -2.0
DEFAULT_LAMBDA_MAX = 2.0
DEFAULT_LAMBDA_STEP = 0.05


def _score_pairs(
    pairs: list[LabeledPair],
    encoder: VoyageNanoEncoder,
    batch_size: int,
    cache_prefix: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Encode both views for a list of pairs, return (s_fwd, s_recip, labels, seeker_look_emb)."""
    seeker_look_texts = [seeker_look_text(lp.pair["userContactFile"], lp.pair["searchQuery"]) for lp in pairs]
    seeker_bg_texts = [bg_text(lp.pair["userContactFile"]) for lp in pairs]
    cand_look_texts = [look_text(lp.pair["matchContactFile"]) for lp in pairs]
    cand_bg_texts = [bg_text(lp.pair["matchContactFile"]) for lp in pairs]

    seeker_look_emb = encoder.encode(
        seeker_look_texts, role="query", batch_size=batch_size, cache_name=f"{cache_prefix}_seeker_look"
    )
    seeker_bg_emb = encoder.encode(
        seeker_bg_texts, role="document", batch_size=batch_size, cache_name=f"{cache_prefix}_seeker_bg"
    )
    cand_look_emb = encoder.encode(
        cand_look_texts, role="query", batch_size=batch_size, cache_name=f"{cache_prefix}_cand_look"
    )
    cand_bg_emb = encoder.encode(
        cand_bg_texts, role="document", batch_size=batch_size, cache_name=f"{cache_prefix}_cand_bg"
    )

    s_fwd = cosine_scores(seeker_look_emb, cand_bg_emb)
    s_recip = cosine_scores(cand_look_emb, seeker_bg_emb)
    labels = np.array([lp.y for lp in pairs], dtype=np.int32)
    return s_fwd, s_recip, labels, seeker_look_emb


def build_bg_corpus(subset_pairs: list[LabeledPair]) -> tuple[list[str], list[str]]:
    """Unique matchContactIds -> narrowed background text (v_i side), first-seen wins."""
    ids: list[str] = []
    texts: list[str] = []
    seen: set[str] = set()
    for lp in subset_pairs:
        record = lp.pair
        mid = record["matchContactId"]
        if mid in seen:
            continue
        seen.add(mid)
        ids.append(mid)
        texts.append(bg_text(record["matchContactFile"]))
    return ids, texts


def eval_real200(
    real_pairs: list[LabeledPair],
    s_fwd: np.ndarray,
    s_recip: np.ndarray,
    seeker_look_emb: np.ndarray,
    lam: float,
    encoder: VoyageNanoEncoder,
    batch_size: int,
) -> dict[str, Any]:
    positives = [p.pair for p in real_pairs if p.label == "pos"]
    negatives = [p.pair for p in real_pairs if p.label == "neg"]
    pos_mask = np.array([p.label == "pos" for p in real_pairs])

    combined = s_fwd + lam * s_recip

    pair_forward_only = pair_metrics(s_fwd[pos_mask], s_fwd[~pos_mask])
    pair_combined = pair_metrics(combined[pos_mask], combined[~pos_mask])

    cand_ids, cand_texts = build_bg_corpus(real_pairs)
    cand_embs = encoder.encode(cand_texts, role="document", batch_size=batch_size, cache_name="real200_bgcorpus")
    pos_query_embs = seeker_look_emb[pos_mask]
    pos_target_ids = [p["matchContactId"] for p in positives]

    retrieval_forward_only = retrieval_metrics(
        query_embs=pos_query_embs,
        target_ids=pos_target_ids,
        candidate_ids=cand_ids,
        candidate_embs=cand_embs,
    )

    neg_seeker_texts = [seeker_to_text(p["userContactFile"], p["searchQuery"]) for p in negatives]
    neg_cand_texts = [candidate_to_text(p["matchContactFile"]) for p in negatives]
    slices_combined = slice_metrics(
        positives=positives,
        negatives=negatives,
        pos_scores=combined[pos_mask],
        neg_scores=combined[~pos_mask],
        neg_seeker_texts=neg_seeker_texts,
        neg_cand_texts=neg_cand_texts,
        query_embs=pos_query_embs,
        target_ids=pos_target_ids,
        candidate_ids=cand_ids,
        candidate_embs=cand_embs,
    )

    return {
        "num_positive": len(positives),
        "num_negative": len(negatives),
        "num_candidates": len(cand_ids),
        "pair_forward_only": pair_forward_only,
        "pair_combined": pair_combined,
        "retrieval_forward_only": retrieval_forward_only,
        "slices_combined": slices_combined,
    }


def run_eval(
    data_dir: Path,
    split_path: Path,
    rrf003_dir: Path,
    model_name: str,
    batch_size: int,
    max_length: int,
    truncate_dim: int | None,
    artifacts_dir: Path,
    lambda_grid: np.ndarray,
) -> dict[str, Any]:
    device = pick_device()
    print(f"device: {device}")
    print(f"model:  {model_name}")

    rrf003 = load_rrf003_pairs(rrf003_dir)
    print(
        f"rrf_003 pairs: {len(rrf003.pairs)} ({rrf003.n_pos} pos / {rrf003.n_neg} neg), "
        f"{rrf003.n_candidates} unique candidates"
    )

    real_all = load_real_pairs(data_dir, split_path, subset="all")
    real_pairs = real_all.pairs
    print(
        f"real pairs: {len(real_pairs)} ({real_all.n_pos} pos / {real_all.n_neg} neg), "
        f"{real_all.n_candidates} unique candidates"
    )

    encoder = VoyageNanoEncoder(
        model_name=model_name,
        device=device,
        max_length=max_length,
        truncate_dim=truncate_dim,
        cache_dir=artifacts_dir,
    )

    # Fit lambda on rrf_003 (judge labels), never touching real labels.
    rrf_fwd, rrf_recip, rrf_labels, _ = _score_pairs(rrf003.pairs, encoder, batch_size, cache_prefix="rrf003")
    lam, rrf_auc_combined = fit_lambda(rrf_fwd, rrf_recip, rrf_labels, lambda_grid)
    rrf_auc_forward_only = float(roc_auc_score(rrf_labels, rrf_fwd))
    print(
        f"fitted lambda={lam} on {len(rrf003.pairs)} rrf_003 pairs "
        f"(rrf_003 AUC: forward-only={rrf_auc_forward_only:.4f}, combined={rrf_auc_combined:.4f})"
    )

    # Freeze lambda, score the real 200 with it.
    real_fwd, real_recip, real_labels, real_seeker_look_emb = _score_pairs(
        real_pairs, encoder, batch_size, cache_prefix="real200"
    )
    real200 = eval_real200(real_pairs, real_fwd, real_recip, real_seeker_look_emb, lam, encoder, batch_size)

    return {
        "model_name": model_name,
        "device": str(device),
        "max_length": max_length,
        "truncate_dim": truncate_dim,
        "batch_size": batch_size,
        "bg_fields": ["positioning", "background"],
        "lambda_fit": {
            "fitted_lambda": lam,
            "fit_population": "rrf_003 (judge-labeled synthetic)",
            "grid_min": float(lambda_grid.min()),
            "grid_max": float(lambda_grid.max()),
            "grid_step": float(lambda_grid[1] - lambda_grid[0]) if len(lambda_grid) > 1 else 0.0,
            "n_rrf003_pairs": len(rrf003.pairs),
            "rrf003_auc_forward_only": rrf_auc_forward_only,
            "rrf003_auc_combined": rrf_auc_combined,
            "rrf003_batch_id": rrf003.batch_id,
            "rrf003_split_hash": rrf003.split_hash,
        },
        "real200": real200,
    }


def print_summary(metrics: dict[str, Any]) -> None:
    lf = metrics["lambda_fit"]
    print("\n=== lambda fit (rrf_003 judge-labeled synthetic pairs only) ===")
    print(f"fitted lambda: {lf['fitted_lambda']}")
    print(f"rrf_003 pairs: {lf['n_rrf003_pairs']}")
    print(f"rrf_003 AUC  forward-only: {lf['rrf003_auc_forward_only']:.4f}")
    print(f"rrf_003 AUC  combined:     {lf['rrf003_auc_combined']:.4f}")

    s = metrics["real200"]
    print(f"\n=== real 200 (frozen lambda, never fit on these labels) ===")
    print(f"({s['num_positive']} pos / {s['num_negative']} neg, {s['num_candidates']} candidates)")
    print(f"pair AUC   forward-only: {s['pair_forward_only']['roc_auc']:.4f}")
    print(f"pair AUC   combined:     {s['pair_combined']['roc_auc']:.4f}")
    print(f"pair AP    forward-only: {s['pair_forward_only']['average_precision']:.4f}")
    print(f"pair AP    combined:     {s['pair_combined']['average_precision']:.4f}")
    r = s["retrieval_forward_only"]
    print(f"retrieval (forward-only ranking): MRR={r['mrr']:.4f} recall@1={r['recall@1']:.4f} ndcg@10={r['ndcg@10']:.4f}")
    hn = s["slices_combined"]["neg_hardness"]
    if hn.get("easy") and hn.get("hard"):
        print(
            f"neg-hardness AUC (combined score): easy={hn['easy'].get('pair_auc')} "
            f"hard={hn['hard'].get('pair_auc')}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Static reciprocal two-tower baseline, lambda calibrated on rrf_003 judge labels"
    )
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--split-path", type=Path, default=Path("data/synthetic/seed_split.json"))
    p.add_argument("--rrf003-dir", type=Path, default=DEFAULT_RRF003_DIR)
    p.add_argument("--model", type=str, default="voyageai/voyage-4-nano")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--truncate-dim", type=int, default=1024)
    p.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/reciprocal_static_rrf003"))
    p.add_argument("--lambda-min", type=float, default=DEFAULT_LAMBDA_MIN)
    p.add_argument("--lambda-max", type=float, default=DEFAULT_LAMBDA_MAX)
    p.add_argument("--lambda-step", type=float, default=DEFAULT_LAMBDA_STEP)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    lambda_grid = build_lambda_grid(args.lambda_min, args.lambda_max, args.lambda_step)

    metrics = run_eval(
        data_dir=args.data_dir,
        split_path=args.split_path,
        rrf003_dir=args.rrf003_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        truncate_dim=args.truncate_dim,
        artifacts_dir=args.artifacts_dir,
        lambda_grid=lambda_grid,
    )
    print_summary(metrics)

    out_path = args.artifacts_dir / "metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
