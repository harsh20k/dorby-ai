"""Real 200-pair evaluation for the trained ask/offer towers.

Same metric shape as baselines/reciprocal_static/eval.py (pair_forward_only,
pair_combined, retrieval_forward_only, slices_combined) so the two are
directly comparable in a results table: reciprocal_static is this score with
zero training on a frozen model, this package is the same score after
jointly training two separate LoRA towers on it. The one difference from
reciprocal_static/eval.py: lambda is not fit here — it was already fixed at
training time (see the training plan's "lambda: fixed, not learned"
section), so this module only scores with the lambda the towers were trained
against, never touching real labels to choose it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from baselines.bert_frozen.text import candidate_to_text, seeker_to_text
from baselines.metrics import pair_metrics, retrieval_metrics, slice_metrics
from baselines.reciprocal_static.text import bg_text, look_text, seeker_look_text
from baselines.voyage_nano.encode import pick_device
from eval_real_full.data import load_real_pairs
from twotower.config import TrainConfig
from twotower.data import LabeledPair
from twotower_ask_offer.model import encode_batched


def build_bg_corpus(subset_pairs: list[LabeledPair]) -> tuple[list[str], list[str]]:
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


def evaluate_ask_offer_pairs(
    ask_model,
    offer_model,
    pairs: list[LabeledPair],
    *,
    lam: float,
    cfg: TrainConfig,
    device: torch.device,
    batch_size: int = 8,
    cache_prefix: str = "eval",
) -> dict[str, Any]:
    """Score `pairs` with the combined S and forward-only s_fwd. Same shape
    as reciprocal_static.eval.eval_subset's return value."""
    seeker_ask_texts = [seeker_look_text(lp.pair["userContactFile"], lp.pair["searchQuery"]) for lp in pairs]
    seeker_offer_texts = [bg_text(lp.pair["userContactFile"]) for lp in pairs]
    cand_ask_texts = [look_text(lp.pair["matchContactFile"]) for lp in pairs]
    cand_offer_texts = [bg_text(lp.pair["matchContactFile"]) for lp in pairs]

    k_seek = encode_batched(ask_model, seeker_ask_texts, role="query", cfg=cfg, device=device, batch_size=batch_size)
    v_seek = encode_batched(offer_model, seeker_offer_texts, role="document", cfg=cfg, device=device, batch_size=batch_size)
    k_cand = encode_batched(ask_model, cand_ask_texts, role="query", cfg=cfg, device=device, batch_size=batch_size)
    v_cand = encode_batched(offer_model, cand_offer_texts, role="document", cfg=cfg, device=device, batch_size=batch_size)

    s_fwd = np.einsum("ij,ij->i", k_seek, v_cand)
    s_recip = np.einsum("ij,ij->i", k_cand, v_seek)
    combined = s_fwd + lam * s_recip

    labels = np.array([lp.y for lp in pairs], dtype=np.int32)
    pos_mask = labels.astype(bool)

    positives = [p.pair for p in pairs if p.label == "pos"]
    negatives = [p.pair for p in pairs if p.label == "neg"]

    pair_forward_only = pair_metrics(s_fwd[pos_mask], s_fwd[~pos_mask])
    pair_combined = pair_metrics(combined[pos_mask], combined[~pos_mask])

    cand_ids, cand_bg_texts = build_bg_corpus(pairs)
    cand_corpus_emb = encode_batched(offer_model, cand_bg_texts, role="document", cfg=cfg, device=device, batch_size=batch_size)
    pos_query_emb = k_seek[pos_mask]
    pos_target_ids = [p["matchContactId"] for p in positives]

    retrieval_forward_only = retrieval_metrics(
        query_embs=pos_query_emb,
        target_ids=pos_target_ids,
        candidate_ids=cand_ids,
        candidate_embs=cand_corpus_emb,
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
        query_embs=pos_query_emb,
        target_ids=pos_target_ids,
        candidate_ids=cand_ids,
        candidate_embs=cand_corpus_emb,
    )

    return {
        "lam": lam,
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
    adapter_dir: Path,
    *,
    lam: float,
    tower_cfg: TrainConfig,
    batch_size: int = 8,
) -> dict[str, Any]:
    from twotower.train import build_model

    device = pick_device()
    print(f"device: {device}")

    ask_model = build_model(tower_cfg, device)
    ask_model.load_adapter(str(adapter_dir / "ask"))
    offer_model = build_model(tower_cfg, device)
    offer_model.load_adapter(str(adapter_dir / "offer"))
    ask_model.eval()
    offer_model.eval()

    real_all = load_real_pairs(data_dir, split_path, subset="all")
    pairs = real_all.pairs
    print(f"real pairs: {len(pairs)} ({real_all.n_pos} pos / {real_all.n_neg} neg), {real_all.n_candidates} candidates")

    holdout_pairs = [p for p in pairs if p.source == "real_holdout"]

    subsets: dict[str, Any] = {}
    for name, subset_pairs in (("holdout", holdout_pairs), ("all", pairs)):
        subsets[name] = evaluate_ask_offer_pairs(
            ask_model, offer_model, subset_pairs,
            lam=lam, cfg=tower_cfg, device=torch.device(device), batch_size=batch_size, cache_prefix=name,
        )

    return {"adapter_dir": str(adapter_dir), "lam": lam, "subsets": subsets}


def print_summary(metrics: dict[str, Any]) -> None:
    print(f"\nlambda (fixed at training time): {metrics['lam']}")
    for name in ("holdout", "all"):
        s = metrics["subsets"][name]
        print(f"\n=== {name} ({s['num_positive']} pos / {s['num_negative']} neg, {s['num_candidates']} candidates) ===")
        print(f"pair AUC   forward-only: {s['pair_forward_only']['roc_auc']:.4f}")
        print(f"pair AUC   combined:     {s['pair_combined']['roc_auc']:.4f}")
        r = s["retrieval_forward_only"]
        print(f"retrieval (forward-only ranking): MRR={r['mrr']:.4f} recall@1={r['recall@1']:.4f}")
        hn = s["slices_combined"]["neg_hardness"]
        if hn.get("easy") and hn.get("hard"):
            print(f"neg-hardness AUC (combined): easy={hn['easy'].get('pair_auc')} hard={hn['hard'].get('pair_auc')}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--split-path", type=Path, default=Path("data/synthetic/seed_split.json"))
    p.add_argument("--adapter-dir", type=Path, required=True)
    p.add_argument("--lam", type=float, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from twotower_ask_offer.config import TOWER_CONFIG

    args = parse_args(argv)
    metrics = run_eval(
        args.data_dir, args.split_path, args.adapter_dir,
        lam=args.lam, tower_cfg=TOWER_CONFIG, batch_size=args.batch_size,
    )
    print_summary(metrics)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(metrics, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
