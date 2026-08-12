"""Holdout eval: frozen Voyage-4-nano cosine ACCEPT vs REJECT on locked B-data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

from baselines.metrics import neg_hardness_slice_metrics, pair_metrics

from bdata_voyage_nano.config import DEFAULT_CONFIG, ExperimentConfig
from bdata_voyage_nano.data import (
    BPair,
    load_pairs,
    load_split,
    partition_pairs,
    within_seeker_dual_label_groups,
)
from bdata_voyage_nano.encode import VoyageNanoEncoder, cosine_scores, pick_device


# Frozen reference numbers from prior voyage_nano runs on the seed dataset
# (artifacts left untouched — quoted here for the comparison table only).
SEED_COMPARISON = {
    "voyage_nano_full_200": {
        "artifact": "artifacts/voyage_nano/metrics.json",
        "population": "200 real seed pairs (100 pos / 100 neg)",
        "pair_auc": 0.5614,
        "hard_neg_auc": 0.5064,
    },
    "voyage_nano_holdout_69": {
        "artifact": "artifacts/voyage_nano_holdout/metrics.json",
        "population": "69-pair frozen holdout (29 pos / 40 neg)",
        "pair_auc": 0.5793103448275863,
        "hard_neg_auc": 0.5706896551724138,
        "easy_neg_auc": 0.6206896551724138,
        "mean_cosine_gap": 0.014442205429077148,
        "max_length_note": "holdout artifact used max_length=4096; this run uses 8192",
    },
    "bdata_tfidf_matched_holdout": {
        "artifact": "artifacts/bdata_tfidf/metrics.json",
        "population": "same B-data seeker-disjoint holdout (split_hash 0f050493daf9)",
        "pair_auc": 0.5120662377540577,
        "hard_neg_auc": 0.40257698493931865,
        "easy_neg_auc": 0.6582453575084077,
        "within_seeker_mean_auc": 0.4970473149429571,
    },
}


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


def encode_aligned(
    encoder: VoyageNanoEncoder,
    texts: Sequence[str],
    *,
    role: Literal["query", "document"],
    batch_size: int,
    cache_name: str,
) -> np.ndarray:
    """Encode texts with per-unique dedup so repeated profiles aren't re-embedded."""
    texts_list = list(texts)
    if not texts_list:
        return encoder.encode([], role=role, batch_size=batch_size, cache_name=cache_name)

    unique: list[str] = []
    index: dict[str, int] = {}
    for t in texts_list:
        if t not in index:
            index[t] = len(unique)
            unique.append(t)

    unique_emb = encoder.encode(
        unique,
        role=role,
        batch_size=batch_size,
        cache_name=f"{cache_name}_u{len(unique)}",
        show_progress=True,
    )
    rows = [index[t] for t in texts_list]
    return unique_emb[np.asarray(rows, dtype=np.int64)]


def run_eval(cfg: ExperimentConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DEFAULT_CONFIG
    device = pick_device()
    print(f"device: {device}")
    print(
        f"model:  {cfg.model_name} "
        f"(max_length={cfg.max_length}, truncate_dim={cfg.truncate_dim}, "
        f"batch_size={cfg.batch_size})"
    )

    pairs, meta = load_pairs(cfg)
    print(
        f"loaded {meta['n_resolved_pairs']} resolved pairs "
        f"({meta['n_accept']} ACCEPT / {meta['n_reject']} REJECT) "
        f"from {meta['n_unique_seekers']} seekers "
        f"(source rows={meta['n_rows']})"
    )

    split = load_split(cfg.split_path)
    _train_pairs, holdout_pairs = partition_pairs(pairs, split)
    print(
        f"split: holdout {len(holdout_pairs)} pairs / {split['n_seekers_holdout']} seekers "
        f"(train pairs unused — frozen encoder)"
    )

    hold_pos, hold_neg = _split_by_label(holdout_pairs)
    print(f"holdout labels: {len(hold_pos)} ACCEPT / {len(hold_neg)} REJECT")

    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    encoder = VoyageNanoEncoder(
        model_name=cfg.model_name,
        device=device,
        max_length=cfg.max_length,
        truncate_dim=cfg.truncate_dim,
        cache_dir=cfg.artifacts_dir,
    )

    tag = split["split_hash"][:12]

    hold_pos_s = encode_aligned(
        encoder,
        [p.seeker_text for p in hold_pos],
        role="query",
        batch_size=cfg.batch_size,
        cache_name=f"{tag}_hold_pos_seeker",
    )
    hold_pos_c = encode_aligned(
        encoder,
        [p.cand_text for p in hold_pos],
        role="document",
        batch_size=cfg.batch_size,
        cache_name=f"{tag}_hold_pos_cand",
    )
    hold_neg_s = encode_aligned(
        encoder,
        [p.seeker_text for p in hold_neg],
        role="query",
        batch_size=cfg.batch_size,
        cache_name=f"{tag}_hold_neg_seeker",
    )
    hold_neg_c = encode_aligned(
        encoder,
        [p.cand_text for p in hold_neg],
        role="document",
        batch_size=cfg.batch_size,
        cache_name=f"{tag}_hold_neg_cand",
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

    return {
        "model_name": cfg.model_name,
        "device": str(device),
        "max_length": cfg.max_length,
        "truncate_dim": cfg.truncate_dim,
        "batch_size": cfg.batch_size,
        "source": meta,
        "split": {
            "path": str(cfg.split_path),
            "split_hash": split["split_hash"],
            "matched_bdata_tfidf_split": True,
            "n_seekers_train": split["n_seekers_train"],
            "n_seekers_holdout": split["n_seekers_holdout"],
            "n_holdout_pairs": len(holdout_pairs),
            "n_holdout_accept": len(hold_pos),
            "n_holdout_reject": len(hold_neg),
        },
        "pair": pair,
        "slices": {"neg_hardness": hardness},
        "within_seeker": within,
        "comparison": SEED_COMPARISON,
        "notes": {
            "label": "ACCEPT=positive, REJECT=negative; PENDING dropped",
            "encoder": "frozen voyage-4-nano; no training; train split unused",
            "text_packing": "baselines.bert_frozen.text seeker_to_text / candidate_to_text",
            "retrieval": "skipped — candidate ids are profile hashes with intentional collisions",
        },
    }


def print_bdata_metrics(metrics: dict[str, Any]) -> None:
    pair = metrics["pair"]
    print("\n=== Pair metrics (B-data holdout) ===")
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
    print(f"n pos/neg:            {pair['num_positive']} / {pair['num_negative']}")

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

    print("\n=== Comparison (prior runs, quoted) ===")
    for key, ref in (metrics.get("comparison") or {}).items():
        print(
            f"{key}: pair_auc={ref.get('pair_auc')} "
            f"hard_neg_auc={ref.get('hard_neg_auc')}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="B-data Voyage-4-nano Accept/Reject (isolated experiment)"
    )
    p.add_argument("--source", type=Path, default=None)
    p.add_argument("--split-path", type=Path, default=None)
    p.add_argument("--artifacts-dir", type=Path, default=None)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--truncate-dim", type=int, default=None)
    return p.parse_args(argv)


def _cfg_from_args(args: argparse.Namespace) -> ExperimentConfig:
    base = DEFAULT_CONFIG
    return ExperimentConfig(
        source_path=args.source or base.source_path,
        split_path=args.split_path or base.split_path,
        artifacts_dir=args.artifacts_dir or base.artifacts_dir,
        model_name=args.model or base.model_name,
        batch_size=args.batch_size if args.batch_size is not None else base.batch_size,
        max_length=args.max_length if args.max_length is not None else base.max_length,
        truncate_dim=(
            args.truncate_dim if args.truncate_dim is not None else base.truncate_dim
        ),
        min_within_seeker_n=base.min_within_seeker_n,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = _cfg_from_args(args)

    metrics = run_eval(cfg)
    print_bdata_metrics(metrics)
    out_path = cfg.artifacts_dir / "metrics.json"
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
