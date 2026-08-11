"""Static reciprocal two-tower baseline (Wu, "Dynamic Reciprocal User Matching
with Fast Weight Programmers", Boardy AI, 2026-06-07).

The paper's main contribution is a Fast Weight Programmer memory that adapts
a user's look-for vector q_t over a session's (impression, skip, like, ...,
timestamp) history (its Section 2). This repo's real 200-pair dataset has no
such data — every pair is a single static (seeker, candidate, accept/decline)
outcome, no session, no per-user interaction log (see data/dataset_summary.md
and docs/objective.md). That mechanism cannot be built here without
fabricating interaction histories, so this experiment does not attempt it.

What the paper's own math reduces to with zero history (its eq. 121,
"q_t = Norm(k_u)") is exactly the population this repo has, so this
experiment implements that slice plus the piece nothing else in this repo has
tried: the static reciprocal term (eqs. 3, 8, 9).

    k_i = E_look(look-for text)      v_i = E_bg(background text)
    s_forward(u, i)   = k_u . v_i    (what the seeker wants vs. candidate offers)
    s_reciprocal(i, u) = k_i . v_u   (what the candidate wants vs. seeker offers)
    S(u, i) = s_forward(u, i) + lambda * s_reciprocal(i, u)

E_look and E_bg are the *same* frozen voyage-4-nano model (no fine-tuning,
matching every other frozen baseline in baselines/) applied to two disjoint
text views per baselines/reciprocal_static/text.py — not the untied,
separately-trained encoders the paper leaves open as "potentially different";
that is a further step, not this one.

lambda is a single scalar fit by 1-D grid search maximizing pair ROC-AUC on
the frozen real TRAIN subset only (eval_real_full, subset="train"), then
applied unchanged to the holdout and all-200 subsets — no eval-subset label
touches fitting, matching the train/holdout discipline used everywhere else
in this repo.

Usage:
  python -m baselines.reciprocal_static.eval --data-dir data
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
from baselines.reciprocal_static.text import bg_text, look_text, seeker_look_text
from baselines.voyage_nano.encode import VoyageNanoEncoder, cosine_scores, pick_device
from eval_real_full.data import load_real_pairs
from twotower.data import LabeledPair

DEFAULT_LAMBDA_MIN = -2.0
DEFAULT_LAMBDA_MAX = 2.0
DEFAULT_LAMBDA_STEP = 0.05


def build_lambda_grid(lo: float, hi: float, step: float) -> np.ndarray:
    return np.round(np.arange(lo, hi + step / 2, step), 4)


def fit_lambda(
    s_fwd: np.ndarray, s_recip: np.ndarray, labels: np.ndarray, lambda_grid: np.ndarray
) -> tuple[float, float]:
    """1-D grid search for lambda maximizing pair ROC-AUC of s_fwd + lambda*s_recip."""
    best_lambda, best_auc = 0.0, float(roc_auc_score(labels, s_fwd))
    for lam in lambda_grid:
        combined = s_fwd + lam * s_recip
        auc = float(roc_auc_score(labels, combined))
        if auc > best_auc:
            best_auc, best_lambda = auc, float(lam)
    return best_lambda, best_auc


def build_bg_corpus(subset_pairs: list[LabeledPair]) -> tuple[list[str], list[str]]:
    """Unique matchContactIds -> background text (v_i side), first-seen wins."""
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


def eval_subset(
    subset_pairs: list[LabeledPair],
    idx_mask: np.ndarray,
    s_fwd: np.ndarray,
    s_recip: np.ndarray,
    seeker_look_emb: np.ndarray,
    lam: float,
    encoder: VoyageNanoEncoder,
    batch_size: int,
    cache_prefix: str,
) -> dict[str, Any]:
    positives = [p.pair for p in subset_pairs if p.label == "pos"]
    negatives = [p.pair for p in subset_pairs if p.label == "neg"]
    pos_mask = np.array([p.label == "pos" for p in subset_pairs])

    fwd_only = s_fwd[idx_mask]
    recip = s_recip[idx_mask]
    combined = fwd_only + lam * recip

    pair_forward_only = pair_metrics(fwd_only[pos_mask], fwd_only[~pos_mask])
    pair_combined = pair_metrics(combined[pos_mask], combined[~pos_mask])

    # Retrieval ranks by forward score only, matching the paper's design: the
    # ANN index holds background embeddings v_i and is queried by k_u alone
    # (Sec. 4.1/4.3); the reciprocal term is a rerank-only signal (Sec. 4.4),
    # never used to build the retrieval ranking itself.
    cand_ids, cand_texts = build_bg_corpus(subset_pairs)
    cand_embs = encoder.encode(
        cand_texts, role="document", batch_size=batch_size, cache_name=f"{cache_prefix}_bgcorpus"
    )
    pos_query_embs = seeker_look_emb[idx_mask][pos_mask]
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

    real_all = load_real_pairs(data_dir, split_path, subset="all")
    pairs = real_all.pairs
    print(
        f"real pairs: {len(pairs)} ({real_all.n_pos} pos / {real_all.n_neg} neg), "
        f"{real_all.n_candidates} unique candidates"
    )

    encoder = VoyageNanoEncoder(
        model_name=model_name,
        device=device,
        max_length=max_length,
        truncate_dim=truncate_dim,
        cache_dir=artifacts_dir,
    )

    seeker_look_texts = [seeker_look_text(lp.pair["userContactFile"], lp.pair["searchQuery"]) for lp in pairs]
    seeker_bg_texts = [bg_text(lp.pair["userContactFile"]) for lp in pairs]
    cand_look_texts = [look_text(lp.pair["matchContactFile"]) for lp in pairs]
    cand_bg_texts = [bg_text(lp.pair["matchContactFile"]) for lp in pairs]

    # k_u: seekers' look-for embedding. Also k_i for the *same* users when they
    # occur as a candidate elsewhere is computed separately below (no query
    # augmentation on that side) — a physical user gets two different k
    # vectors depending on which role a given pair puts them in, matching how
    # every other baseline in this repo already treats seeker vs. candidate
    # text asymmetrically (searchQuery only ever appears on the seeker side).
    seeker_look_emb = encoder.encode(seeker_look_texts, role="query", batch_size=batch_size, cache_name="all_seeker_look")
    seeker_bg_emb = encoder.encode(seeker_bg_texts, role="document", batch_size=batch_size, cache_name="all_seeker_bg")
    cand_look_emb = encoder.encode(cand_look_texts, role="query", batch_size=batch_size, cache_name="all_cand_look")
    cand_bg_emb = encoder.encode(cand_bg_texts, role="document", batch_size=batch_size, cache_name="all_cand_bg")

    s_fwd = cosine_scores(seeker_look_emb, cand_bg_emb)
    s_recip = cosine_scores(cand_look_emb, seeker_bg_emb)
    labels = np.array([lp.y for lp in pairs], dtype=np.int32)

    train_mask = np.array([lp.source == "real_train" for lp in pairs])
    holdout_mask = ~train_mask
    all_mask = np.ones(len(pairs), dtype=bool)

    lam, train_auc_combined = fit_lambda(s_fwd[train_mask], s_recip[train_mask], labels[train_mask], lambda_grid)
    train_auc_forward_only = float(roc_auc_score(labels[train_mask], s_fwd[train_mask]))
    print(
        f"fitted lambda={lam} on {int(train_mask.sum())} train pairs "
        f"(train AUC: forward-only={train_auc_forward_only:.4f}, combined={train_auc_combined:.4f})"
    )

    subsets: dict[str, Any] = {}
    for name, mask in (("holdout", holdout_mask), ("all", all_mask)):
        subset_pairs = [p for p, m in zip(pairs, mask) if m]
        subsets[name] = eval_subset(
            subset_pairs,
            mask,
            s_fwd,
            s_recip,
            seeker_look_emb,
            lam,
            encoder,
            batch_size,
            cache_prefix=name,
        )

    return {
        "model_name": model_name,
        "device": str(device),
        "max_length": max_length,
        "truncate_dim": truncate_dim,
        "batch_size": batch_size,
        "lambda_fit": {
            "fitted_lambda": lam,
            "grid_min": float(lambda_grid.min()),
            "grid_max": float(lambda_grid.max()),
            "grid_step": float(lambda_grid[1] - lambda_grid[0]) if len(lambda_grid) > 1 else 0.0,
            "n_train_pairs": int(train_mask.sum()),
            "train_auc_forward_only": train_auc_forward_only,
            "train_auc_combined": train_auc_combined,
        },
        "subsets": subsets,
    }


def print_summary(metrics: dict[str, Any]) -> None:
    lf = metrics["lambda_fit"]
    print("\n=== lambda fit (real TRAIN subset only) ===")
    print(f"fitted lambda: {lf['fitted_lambda']}")
    print(f"train pairs:   {lf['n_train_pairs']}")
    print(f"train AUC  forward-only: {lf['train_auc_forward_only']:.4f}")
    print(f"train AUC  combined:     {lf['train_auc_combined']:.4f}")
    for name in ("holdout", "all"):
        s = metrics["subsets"][name]
        print(f"\n=== {name} ({s['num_positive']} pos / {s['num_negative']} neg, {s['num_candidates']} candidates) ===")
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
    p = argparse.ArgumentParser(description="Static reciprocal two-tower baseline (Wu FWP paper, cold-start slice)")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--split-path", type=Path, default=Path("data/synthetic/seed_split.json"))
    p.add_argument("--model", type=str, default="voyageai/voyage-4-nano")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--truncate-dim", type=int, default=1024)
    p.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/reciprocal_static"))
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
