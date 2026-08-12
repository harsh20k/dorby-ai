"""Holdout eval: TF-IDF cosine ACCEPT vs REJECT on locked B-data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from baselines.metrics import neg_hardness_slice_metrics, pair_metrics

from bdata_tfidf.config import DEFAULT_CONFIG, ExperimentConfig
from bdata_tfidf.data import (
    BPair,
    build_seeker_disjoint_split,
    load_pairs,
    load_split,
    partition_pairs,
    within_seeker_dual_label_groups,
    write_split,
)
from bdata_tfidf.encode import TfidfEncoder, cosine_scores, pick_device


def _split_by_label(pairs: list[BPair]) -> tuple[list[BPair], list[BPair]]:
    pos = [p for p in pairs if p.label == "ACCEPT"]
    neg = [p for p in pairs if p.label == "REJECT"]
    return pos, neg


def _within_seeker_auc(pairs: list[BPair], scores_by_id: dict[str, float]) -> dict[str, Any]:
    groups = within_seeker_dual_label_groups(pairs)
    per_seeker: list[float] = []
    for _sid, g in groups.items():
        y = [1] * len(g["ACCEPT"]) + [0] * len(g["REJECT"])
        s = [scores_by_id[p.pair_id] for p in g["ACCEPT"] + g["REJECT"]]
        if len(set(y)) < 2:
            continue
        try:
            per_seeker.append(float(roc_auc_score(y, s)))
        except ValueError:
            continue
    if not per_seeker:
        return {
            "n_seekers": 0,
            "mean_auc": None,
            "skipped": "no dual-label seekers",
        }
    return {
        "n_seekers": len(per_seeker),
        "mean_auc": float(np.mean(per_seeker)),
        "median_auc": float(np.median(per_seeker)),
        "min_auc": float(np.min(per_seeker)),
        "max_auc": float(np.max(per_seeker)),
    }


def run_eval(cfg: ExperimentConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DEFAULT_CONFIG
    device = pick_device()
    ngram = (cfg.ngram_min, cfg.ngram_max)
    print(f"device: {device}")
    print(
        f"model:  bdata_tfidf(max_features={cfg.max_features}, ngram_range={ngram})"
    )

    pairs, meta = load_pairs(cfg)
    print(
        f"loaded {meta['n_resolved_pairs']} resolved pairs "
        f"({meta['n_accept']} ACCEPT / {meta['n_reject']} REJECT) "
        f"from {meta['n_unique_seekers']} seekers "
        f"(source rows={meta['n_rows']})"
    )

    split = load_split(cfg.split_path)
    train_pairs, holdout_pairs = partition_pairs(pairs, split)
    print(
        f"split: train {len(train_pairs)} pairs / {split['n_seekers_train']} seekers | "
        f"holdout {len(holdout_pairs)} pairs / {split['n_seekers_holdout']} seekers"
    )

    train_pos, train_neg = _split_by_label(train_pairs)
    hold_pos, hold_neg = _split_by_label(holdout_pairs)
    print(
        f"holdout labels: {len(hold_pos)} ACCEPT / {len(hold_neg)} REJECT"
    )

    # Fit vocabulary/IDF on train texts only (seeker + candidate).
    train_texts = [p.seeker_text for p in train_pairs] + [p.cand_text for p in train_pairs]
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    encoder = TfidfEncoder(
        max_features=cfg.max_features,
        ngram_range=ngram,
        min_df=cfg.min_df,
        cache_dir=cfg.artifacts_dir,
    )
    encoder.fit(train_texts)
    print(f"fitted TF-IDF on {len(train_texts)} train texts")

    tag = split["split_hash"][:12]

    hold_pos_s = encoder.encode(
        [p.seeker_text for p in hold_pos], cache_name=f"{tag}_hold_pos_seeker"
    )
    hold_pos_c = encoder.encode(
        [p.cand_text for p in hold_pos], cache_name=f"{tag}_hold_pos_cand"
    )
    hold_neg_s = encoder.encode(
        [p.seeker_text for p in hold_neg], cache_name=f"{tag}_hold_neg_seeker"
    )
    hold_neg_c = encoder.encode(
        [p.cand_text for p in hold_neg], cache_name=f"{tag}_hold_neg_cand"
    )

    pos_scores = cosine_scores(hold_pos_s, hold_pos_c)
    neg_scores = cosine_scores(hold_neg_s, hold_neg_c)
    pair = pair_metrics(pos_scores, neg_scores)

    hardness = neg_hardness_slice_metrics(
        neg_scores=neg_scores,
        neg_seeker_texts=[p.seeker_text for p in hold_neg],
        neg_cand_texts=[p.cand_text for p in hold_neg],
        pos_scores=pos_scores,
    )

    scores_by_id = {
        **{p.pair_id: float(s) for p, s in zip(hold_pos, pos_scores)},
        **{p.pair_id: float(s) for p, s in zip(hold_neg, neg_scores)},
    }
    within = _within_seeker_auc(holdout_pairs, scores_by_id)
    if within["n_seekers"] < cfg.min_within_seeker_n:
        within = {
            **within,
            "reported": False,
            "note": (
                f"n_seekers={within['n_seekers']} < min_within_seeker_n="
                f"{cfg.min_within_seeker_n}; metric recorded but not headline"
            ),
        }
    else:
        within = {**within, "reported": True}

    # Train-set pair metrics for leakage/overfit sanity (fit set, not a decision gate)
    train_pos_s = encoder.encode(
        [p.seeker_text for p in train_pos], cache_name=f"{tag}_train_pos_seeker"
    )
    train_pos_c = encoder.encode(
        [p.cand_text for p in train_pos], cache_name=f"{tag}_train_pos_cand"
    )
    train_neg_s = encoder.encode(
        [p.seeker_text for p in train_neg], cache_name=f"{tag}_train_neg_seeker"
    )
    train_neg_c = encoder.encode(
        [p.cand_text for p in train_neg], cache_name=f"{tag}_train_neg_cand"
    )
    train_pair = pair_metrics(
        cosine_scores(train_pos_s, train_pos_c),
        cosine_scores(train_neg_s, train_neg_c),
    )

    return {
        "model_name": f"bdata_tfidf(max_features={cfg.max_features},ngram_range={list(ngram)})",
        "device": device,
        "max_features": cfg.max_features,
        "ngram_range": list(ngram),
        "source": meta,
        "split": {
            "path": str(cfg.split_path),
            "split_hash": split["split_hash"],
            "n_seekers_train": split["n_seekers_train"],
            "n_seekers_holdout": split["n_seekers_holdout"],
            "n_train_pairs": len(train_pairs),
            "n_holdout_pairs": len(holdout_pairs),
            "n_holdout_accept": len(hold_pos),
            "n_holdout_reject": len(hold_neg),
        },
        "pair": pair,
        "pair_train_sanity": train_pair,
        "slices": {"neg_hardness": hardness},
        "within_seeker": within,
        "notes": {
            "label": "ACCEPT=positive, REJECT=negative; PENDING dropped",
            "fit": "TF-IDF fitted on train seeker+candidate texts only",
            "retrieval": "skipped — candidate ids are profile hashes with intentional collisions",
        },
    }


def print_bdata_metrics(metrics: dict[str, Any]) -> None:
    pair = metrics["pair"]
    print("\n=== Pair metrics (holdout) ===")
    print(f"ROC-AUC:              {pair['roc_auc']:.4f}")
    print(f"Average Precision:    {pair['average_precision']:.4f}")
    print(
        f"Best-F1:              {pair['best_f1']:.4f} "
        f"@ threshold={pair['best_f1_threshold']:.4f} "
        f"(acc={pair['best_f1_accuracy']:.4f})"
    )
    print(f"Accuracy @ 0.5:       {pair['accuracy_at_0.5']:.4f}")
    print(
        f"Mean cosine (pos/neg/gap): "
        f"{pair['mean_cosine_positive']:.4f} / "
        f"{pair['mean_cosine_negative']:.4f} / "
        f"{pair['mean_cosine_gap']:.4f}"
    )
    print(
        f"n pos/neg:            {pair['num_positive']} / {pair['num_negative']}"
    )

    train = metrics.get("pair_train_sanity") or {}
    if train:
        print("\n=== Pair metrics (train sanity) ===")
        print(f"ROC-AUC:              {train['roc_auc']:.4f}")

    hard = (metrics.get("slices") or {}).get("neg_hardness") or {}
    easy = hard.get("easy") or {}
    hard_b = hard.get("hard") or {}
    if easy or hard_b:
        print("\n=== Neg-hardness slices ===")
        if easy.get("pair_auc") is not None:
            print(
                f"easy-neg AUC:         {easy['pair_auc']:.4f} "
                f"(n_neg={easy.get('n_negatives')})"
            )
        if hard_b.get("pair_auc") is not None:
            print(
                f"hard-neg AUC:         {hard_b['pair_auc']:.4f} "
                f"(n_neg={hard_b.get('n_negatives')})"
            )

    within = metrics.get("within_seeker") or {}
    print("\n=== Within-seeker ranking ===")
    if within.get("mean_auc") is None:
        print(f"skipped: {within.get('skipped') or within.get('note')}")
    else:
        flag = "headline" if within.get("reported") else "informational only"
        print(
            f"mean AUC:             {within['mean_auc']:.4f} "
            f"(n_seekers={within['n_seekers']}, {flag})"
        )


def init_split(cfg: ExperimentConfig | None = None, *, force: bool = False) -> Path:
    cfg = cfg or DEFAULT_CONFIG
    if cfg.split_path.exists() and not force:
        raise FileExistsError(
            f"{cfg.split_path} already exists. Pass --force to overwrite."
        )
    pairs, meta = load_pairs(cfg)
    split = build_seeker_disjoint_split(
        pairs, holdout_frac=cfg.holdout_frac, seed=cfg.split_seed
    )
    split["source_sha256"] = meta["provenance"]["sha256"]
    split["source_size_bytes"] = meta["provenance"]["size_bytes"]
    write_split(split, cfg.split_path)
    print(f"Wrote {cfg.split_path}")
    print(
        f"seekers train/holdout: {split['n_seekers_train']}/{split['n_seekers_holdout']} | "
        f"pairs train/holdout: {split['n_train_pairs']}/{split['n_holdout_pairs']}"
    )
    print(f"split_hash: {split['split_hash']}")
    return cfg.split_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="B-data TF-IDF Accept/Reject (isolated experiment)"
    )
    p.add_argument(
        "--init-split",
        action="store_true",
        help="Freeze seeker-disjoint train/holdout split to bdata_tfidf/split.json",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="With --init-split, overwrite an existing split.json",
    )
    p.add_argument("--source", type=Path, default=None)
    p.add_argument("--split-path", type=Path, default=None)
    p.add_argument("--artifacts-dir", type=Path, default=None)
    p.add_argument("--max-features", type=int, default=None)
    p.add_argument("--ngram-min", type=int, default=None)
    p.add_argument("--ngram-max", type=int, default=None)
    return p.parse_args(argv)


def _cfg_from_args(args: argparse.Namespace) -> ExperimentConfig:
    base = DEFAULT_CONFIG
    return ExperimentConfig(
        source_path=args.source or base.source_path,
        split_path=args.split_path or base.split_path,
        artifacts_dir=args.artifacts_dir or base.artifacts_dir,
        max_features=args.max_features if args.max_features is not None else base.max_features,
        ngram_min=args.ngram_min if args.ngram_min is not None else base.ngram_min,
        ngram_max=args.ngram_max if args.ngram_max is not None else base.ngram_max,
        min_df=base.min_df,
        holdout_frac=base.holdout_frac,
        split_seed=base.split_seed,
        min_within_seeker_n=base.min_within_seeker_n,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = _cfg_from_args(args)
    if args.init_split:
        init_split(cfg, force=args.force)
        return 0

    metrics = run_eval(cfg)
    print_bdata_metrics(metrics)
    out_path = cfg.artifacts_dir / "metrics.json"
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
