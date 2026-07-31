"""Fit the sectioned MoE on pairs, with the loss applied at pair level.

**The label lives on the pair, the rows are sections.** That makes this multiple-
instance learning: the bag (pair) is labeled, the instances (asks) are not. So
the loss is computed on the *pooled* pair logit, and gradient reaches the section
network only through the pooling weights. This is the mechanism by which the
model is supposed to discover which ask carried a pair — nothing supervises that
directly.

The two gate-entropy terms are applied at row level, where routing happens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from .config import SectionedConfig
from .model import SectionedMoE, balance_loss, sharpen_loss


@dataclass
class FitResult:
    model: SectionedMoE
    final_loss: float
    epochs_run: int


def _tensors(
    sim: np.ndarray, interaction: np.ndarray, section_emb: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(np.ascontiguousarray(sim)),
        torch.from_numpy(np.ascontiguousarray(interaction)),
        torch.from_numpy(np.ascontiguousarray(section_emb)),
    )


def fit(
    cfg: SectionedConfig,
    *,
    sim: np.ndarray,
    interaction: np.ndarray,
    section_emb: np.ndarray,
    groups: Sequence[np.ndarray],
    pair_labels: np.ndarray,
    model: SectionedMoE | None = None,
    epochs: int | None = None,
    seed: int | None = None,
) -> FitResult:
    """Train on whole pairs. ``model`` may be passed in to continue from a warm start."""
    torch.manual_seed(cfg.seed if seed is None else seed)

    m = model or SectionedMoE(
        n_sim=sim.shape[1],
        emb_dim=section_emb.shape[1],
        n_experts=cfg.n_experts,
        expert_hidden=cfg.expert_hidden,
        expert_out=cfg.expert_out,
        interaction_dims=cfg.interaction_dims,
        gate_dims=cfg.gate_dims,
        tau=cfg.tau,
        expert_dropout=cfg.expert_dropout,
        pooling=cfg.pooling,
        pool_tau=cfg.pool_tau,
    )

    t_sim, t_int, t_emb = _tensors(sim, interaction, section_emb)
    t_groups = [torch.from_numpy(g) for g in groups]
    y = torch.from_numpy(pair_labels.astype(np.float32))

    opt = torch.optim.AdamW(m.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    bce = torch.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(cfg.seed if seed is None else seed)

    n_pairs = len(t_groups)
    n_epochs = cfg.epochs if epochs is None else epochs
    last = float("nan")

    for _ in range(n_epochs):
        m.train()
        order = rng.permutation(n_pairs)
        for start in range(0, n_pairs, cfg.batch_pairs):
            batch = order[start : start + cfg.batch_pairs]
            if len(batch) < 2:
                continue
            # Gather this batch's rows, then renumber the groups onto them.
            row_idx = torch.cat([t_groups[p] for p in batch])
            local, cursor = [], 0
            for p in batch:
                n = len(t_groups[p])
                local.append(torch.arange(cursor, cursor + n))
                cursor += n

            pair_logits, gates, _ = m(
                t_sim[row_idx], t_int[row_idx], t_emb[row_idx], local
            )
            loss = (
                bce(pair_logits, y[batch])
                + cfg.sharpen_weight * sharpen_loss(gates)
                + cfg.balance_weight * balance_loss(gates)
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            last = float(loss.detach())

    return FitResult(model=m, final_loss=last, epochs_run=n_epochs)


@torch.no_grad()
def predict(
    m: SectionedMoE,
    *,
    sim: np.ndarray,
    interaction: np.ndarray,
    section_emb: np.ndarray,
    groups: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """(pair scores, gate weights per row, pooling weights per pair)."""
    m.eval()
    t_sim, t_int, t_emb = _tensors(sim, interaction, section_emb)
    t_groups = [torch.from_numpy(g) for g in groups]
    pair_logits, gates, weights = m(t_sim, t_int, t_emb, t_groups)
    return (
        torch.sigmoid(pair_logits).numpy(),
        gates.numpy(),
        [w.numpy() for w in weights],
    )
