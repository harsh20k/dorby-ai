"""Reciprocal in-batch-negative loss: S = s_fwd + lambda*s_rev, cross-entropy
on the diagonal.

Same shape as MultipleNegativesRankingLoss (today's recipe, see
twotower_voyage_gemini_ctrl/train.py) — a positive/negative pool built once
per batch, the true pair on the diagonal, every other cell a free in-batch
negative — generalized from a single cosine-similarity matrix to the combined
reciprocal score matrix S. This is the one piece of the whole package that is
genuinely new math (everything else is plumbing); `combine_and_cross_entropy`
is kept separate from the encoding step specifically so it can be unit-tested
on plain tensors, no model or GPU required.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from twotower_ask_offer.data import AskOfferRow
from baselines.reciprocal_static.text import bg_text, look_text, seeker_look_text


@dataclass(frozen=True)
class BatchTexts:
    seeker_ask: list[str]  # k_u source text (lookingFor + query)
    seeker_offer: list[str]  # v_u source text (all fields except lookingFor)
    pool_ask: list[str]  # k_i source text, positives then negatives
    pool_offer: list[str]  # v_i source text, positives then negatives
    n: int  # batch size (number of anchors)


def build_batch_texts(rows: list[AskOfferRow]) -> BatchTexts:
    n = len(rows)
    seeker_ask = [seeker_look_text(r.seeker_profile, r.search_query) for r in rows]
    seeker_offer = [bg_text(r.seeker_profile) for r in rows]

    pool_ask = [look_text(r.positive_profile) for r in rows]
    pool_offer = [bg_text(r.positive_profile) for r in rows]

    n_negatives = len(rows[0].negative_profiles) if rows else 0
    for neg_idx in range(n_negatives):
        for r in rows:
            neg_profile = r.negative_profiles[neg_idx]
            pool_ask.append(look_text(neg_profile))
            pool_offer.append(bg_text(neg_profile))

    return BatchTexts(seeker_ask, seeker_offer, pool_ask, pool_offer, n=n)


def combine_and_cross_entropy(
    k_seek: torch.Tensor,
    v_seek: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    *,
    lam: float,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure tensor math, no model calls — testable on fake embeddings.

    k_seek, v_seek: [N, d] — seeker's ask/offer embeddings.
    k_pool, v_pool: [P, d] — pool's ask/offer embeddings, true pair for row i
        at column i (P >= N, positives first).

    s_fwd[i,j] = k_seek[i] . v_pool[j]   (does pool item j suit seeker i)
    s_rev[i,j] = v_seek[i] . k_pool[j]   (does seeker i suit pool item j)
    S = s_fwd + lam * s_rev

    Returns (loss, S) where S is the raw (unscaled) combined-score matrix —
    callers use S's diagonal for pair-level metrics, argmax(S, dim=1) for
    batch/dev accuracy.
    """
    if k_seek.shape != v_seek.shape:
        raise ValueError(f"seeker ask/offer shape mismatch: {k_seek.shape} vs {v_seek.shape}")
    if k_pool.shape != v_pool.shape:
        raise ValueError(f"pool ask/offer shape mismatch: {k_pool.shape} vs {v_pool.shape}")
    n, p = k_seek.shape[0], k_pool.shape[0]
    if p < n:
        raise ValueError(f"pool size {p} smaller than batch size {n} — diagonal labels would be invalid")

    s_fwd = k_seek @ v_pool.T
    s_rev = v_seek @ k_pool.T
    s = s_fwd + lam * s_rev

    labels = torch.arange(n, device=s.device)
    loss = F.cross_entropy(s * scale, labels)
    return loss, s
