"""Dev evaluator for checkpoint selection: ranks each dev seeker against the
full dev corpus using the combined score S, not a single embedding's cosine
similarity — the ranking equivalent of the training loss.

Corpus construction and rank->metrics conversion mirror
`twotower_voyage_gemini_ctrl/eval_dev.py::CorpusRecallDevEvaluator` (same
"rank against every unique dev candidate" design, same reasoning: a triplet
test over 2-3 candidates does not predict what a ~178-candidate holdout
ranking will do). `retrieval_metrics_from_ranks` (baselines/metrics.py) is
reused unchanged so dev and holdout compute recall@1 identically by
construction — the one difference from that file's evaluator is the score
itself: S = s_fwd + lambda*s_rev instead of plain cosine, since that's what
this experiment's loss actually optimizes and checkpoint selection should
track the same quantity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from baselines.metrics import retrieval_metrics_from_ranks
from twotower_ask_offer_posbg.text import bg_text, look_text, seeker_look_text
from twotower.config import TrainConfig
from twotower_ask_offer_posbg.data import AskOfferRow
from twotower_ask_offer_posbg.model import encode_batched


def build_dev_corpus(rows: Sequence[AskOfferRow]) -> tuple[list[str], list[Any]]:
    """Unique (candidate_id, profile) across every positive and negative in
    the dev split. Positives inserted first, matching the voyage_gemini_ctrl
    evaluator's tie-break rule."""
    ids: list[str] = []
    profiles: list[Any] = []
    seen: set[str] = set()
    for row in rows:
        if row.positive_id not in seen:
            seen.add(row.positive_id)
            ids.append(row.positive_id)
            profiles.append(row.positive_profile)
    for row in rows:
        for cid, profile in zip(row.negative_ids, row.negative_profiles):
            if cid in seen:
                continue
            seen.add(cid)
            ids.append(cid)
            profiles.append(profile)
    return ids, profiles


@torch.no_grad()
def evaluate_dev(
    ask_model: SentenceTransformer,
    offer_model: SentenceTransformer,
    rows: Sequence[AskOfferRow],
    *,
    lam: float,
    cfg: TrainConfig,
    device: torch.device,
    batch_size: int = 8,
    name: str = "train_dev",
    output_path: Path | None = None,
    epoch: int | None = None,
) -> dict[str, float]:
    corpus_ids, corpus_profiles = build_dev_corpus(rows)
    corpus_ask_texts = [look_text(p) for p in corpus_profiles]
    corpus_offer_texts = [bg_text(p) for p in corpus_profiles]

    k_corpus = encode_batched(ask_model, corpus_ask_texts, role="query", cfg=cfg, device=device, batch_size=batch_size)
    v_corpus = encode_batched(offer_model, corpus_offer_texts, role="document", cfg=cfg, device=device, batch_size=batch_size)

    seeker_ask_texts = [seeker_look_text(r.seeker_profile, r.search_query) for r in rows]
    seeker_offer_texts = [bg_text(r.seeker_profile) for r in rows]
    k_seek = encode_batched(ask_model, seeker_ask_texts, role="query", cfg=cfg, device=device, batch_size=batch_size)
    v_seek = encode_batched(offer_model, seeker_offer_texts, role="document", cfg=cfg, device=device, batch_size=batch_size)

    s_fwd = k_seek @ v_corpus.T
    s_rev = v_seek @ k_corpus.T
    s = s_fwd + lam * s_rev

    id_to_idx = {cid: i for i, cid in enumerate(corpus_ids)}
    ranks: list[int] = []
    for row_idx, row in enumerate(rows):
        target_idx = id_to_idx[row.positive_id]
        order = np.argsort(-s[row_idx], kind="stable")
        rank = int(np.where(order == target_idx)[0][0]) + 1
        ranks.append(rank)

    metrics = retrieval_metrics_from_ranks(ranks) if ranks else {}
    flat = {f"{name}_{k}": float(v) for k, v in metrics.items()}
    raw = {"n_rows": len(rows), "n_corpus": len(corpus_ids), "lam": lam, "retrieval": metrics}

    print(
        f"  dev[{name}] corpus={len(corpus_ids)} "
        f"R@1={flat.get(f'{name}_recall@1', 0.0):.4f} "
        f"R@10={flat.get(f'{name}_recall@10', 0.0):.4f} "
        f"MRR={flat.get(f'{name}_mrr', 0.0):.4f}"
    )

    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)
        suffix = f"_epoch{epoch}" if epoch is not None else ""
        (output_path / f"{name}_metrics{suffix}.json").write_text(
            json.dumps({"raw": raw, "flat": flat}, indent=2) + "\n"
        )
    return flat
