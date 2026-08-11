"""Sectioned mixture-of-experts with learned attention pooling.

Architecture::

    (pair, section) row
        similarity(3) ─┐
        pair scalars(12)├─> [expert_1 .. expert_M]  each 47 -> 24 -> ReLU -> 16 -> ReLU
        interaction(D) ─┘         (D projected to 32 by a learned layer)
                                        |
    section embedding ──> gate g(x) = softmax(a(x) / tau)   <- routes on the ASK only
                                        |
                        mixed = sum_m g_m * expert_m(row)
                                        |
                              head -> per-section verdict
                                        |
                     pooling over a pair's sections -> pair logit

**Two differences from ``moe_reranker.model.MMoE``, both deliberate.**

1. *The gate sees only the section embedding.* In the pair-level model the gate
   saw the same features as the experts, so "what is this expert for" was
   unconstrained and the diagnostics had to infer it after the fact — and once
   caught routing tracking *seeker identity* rather than anything semantic. Here
   routing is structurally forced to depend on the ask and nothing else, which
   turns that diagnostic from an inference into a direct check.

2. *Experts are two layers, not one.* The earlier expert was
   ``Linear(12->4) + ReLU`` — a weighted sum with a kink, four of which can
   barely out-express one logistic regression. That is a plain reason the
   mixture never beat one. Two layers is affordable here because the model is
   fit on ~1,050 rows rather than 111.

**Pooling is the experiment.** ``attention`` learns which ask decided a pair;
``softmax`` freezes those weights at temperature ``pool_tau``, which is the rule
the section-aggregation sweep already measured as best. Both are implemented so
they can be run back to back on identical data — if learned pooling does not beat
frozen pooling, that is a real and useful negative result, not a failed run.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

POOLING_MODES = ("attention", "softmax", "mean", "max")


class SectionedMoE(nn.Module):
    def __init__(
        self,
        n_sim: int,
        emb_dim: int,
        *,
        n_experts: int = 4,
        expert_hidden: int = 24,
        expert_out: int = 16,
        interaction_dims: int = 32,
        gate_dims: int = 16,
        tau: float = 0.05,
        expert_dropout: float = 0.2,
        pooling: str = "attention",
        pool_tau: float = 0.05,
    ) -> None:
        super().__init__()
        if pooling not in POOLING_MODES:
            raise ValueError(f"pooling must be one of {POOLING_MODES}, got {pooling!r}")
        if tau <= 0 or pool_tau <= 0:
            raise ValueError("temperatures must be > 0")
        if not 0.0 <= expert_dropout < 1.0:
            raise ValueError("expert_dropout must be in [0, 1)")

        self.n_experts = n_experts
        self.tau = tau
        self.pool_tau = pool_tau
        self.pooling = pooling
        self.expert_dropout = expert_dropout

        # The elementwise product is high-dimensional; reduce it *with* the task
        # rather than before it, so the reduction can keep what the label needs.
        self.interaction_proj = nn.Linear(emb_dim, interaction_dims)
        # The gate's view of the ask. Separate from the interaction projection on
        # purpose: routing should describe the ask, not the ask-candidate match.
        self.gate_proj = nn.Linear(emb_dim, gate_dims)

        expert_in = n_sim + interaction_dims
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(expert_in, expert_hidden),
                nn.ReLU(),
                nn.Linear(expert_hidden, expert_out),
                nn.ReLU(),
            )
            for _ in range(n_experts)
        )
        self.gate = nn.Linear(gate_dims, n_experts)
        self.head = nn.Linear(expert_out, 1)
        # One learned query vector. This is the whole of "learned attention" —
        # ~16 parameters deciding which ask speaks for the pair.
        self.attn_query = nn.Parameter(torch.zeros(gate_dims))
        nn.init.normal_(self.attn_query, std=0.1)

    # ------------------------------------------------------------------ parts
    def gate_weights(self, gate_in: torch.Tensor) -> torch.Tensor:
        """(N, n_experts) routing weights, computed from the ask alone."""
        logits = self.gate(gate_in)
        if self.training and self.expert_dropout > 0:
            keep = (
                torch.rand(logits.shape[0], self.n_experts, device=logits.device)
                >= self.expert_dropout
            )
            # Never drop every expert for a row.
            keep = keep | ~keep.any(dim=-1, keepdim=True)
            logits = logits.masked_fill(~keep, float("-inf"))
        return F.softmax(logits / self.tau, dim=-1)

    def section_logits(
        self, sim: torch.Tensor, interaction: torch.Tensor, section_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-section verdicts. Returns (logits (N,), gates (N,M), gate_in (N,G))."""
        inter = torch.relu(self.interaction_proj(interaction))
        x = torch.cat([sim, inter], dim=1)
        gate_in = self.gate_proj(section_emb)
        gates = self.gate_weights(gate_in)

        expert_out = torch.stack([e(x) for e in self.experts], dim=1)  # (N, M, H)
        mixed = (gates.unsqueeze(-1) * expert_out).sum(dim=1)  # (N, H)
        return self.head(mixed).squeeze(-1), gates, gate_in

    def pool(
        self,
        logits: torch.Tensor,
        gate_in: torch.Tensor,
        groups: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Combine each pair's section verdicts into one pair logit.

        Returns (pair_logits (P,), per-pair pooling weights) — the weights are
        returned because "which ask decided this" is the interpretable by-product
        that motivated the design, and it should never be thrown away.
        """
        pair_logits, weights = [], []
        for idx in groups:
            z = logits[idx]
            if z.numel() == 1:
                w = torch.ones(1, device=z.device)
            elif self.pooling == "attention":
                w = F.softmax(gate_in[idx] @ self.attn_query, dim=0)
            elif self.pooling == "softmax":
                # Frozen control: the rule the aggregation sweep already picked.
                w = F.softmax(z / self.pool_tau, dim=0)
            elif self.pooling == "max":
                w = F.one_hot(z.argmax(), num_classes=z.numel()).to(z.dtype)
            else:  # mean
                w = torch.full_like(z, 1.0 / z.numel())
            pair_logits.append((w * z).sum())
            weights.append(w)
        return torch.stack(pair_logits), weights

    def forward(
        self,
        sim: torch.Tensor,
        interaction: torch.Tensor,
        section_emb: torch.Tensor,
        groups: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        sec_logits, gates, gate_in = self.section_logits(sim, interaction, section_emb)
        pair_logits, weights = self.pool(sec_logits, gate_in, groups)
        return pair_logits, gates, weights

    @torch.no_grad()
    def score_pairs(
        self,
        sim: torch.Tensor,
        interaction: torch.Tensor,
        section_emb: torch.Tensor,
        groups: list[torch.Tensor],
    ) -> torch.Tensor:
        self.eval()
        pair_logits, _, _ = self.forward(sim, interaction, section_emb, groups)
        return torch.sigmoid(pair_logits)


# --------------------------------------------------------------------- losses
def sharpen_loss(gates: torch.Tensor) -> torch.Tensor:
    """Mean per-example gate entropy. **Minimize** so each ask commits.

    Alone this collapses everything onto one expert, which is why it is always
    paired with ``balance_loss``.
    """
    p = gates.clamp_min(1e-9)
    return -(p * p.log()).sum(dim=-1).mean()


def balance_loss(gates: torch.Tensor) -> torch.Tensor:
    """Negative entropy of the batch-average gate. **Minimize** to keep experts alive.

    Pulls directly against ``sharpen_loss``: that one wants each row decisive,
    this one wants the population spread. Both are needed — the pair of them is
    "strong opinions, held by different experts".
    """
    avg = gates.mean(dim=0).clamp_min(1e-9)
    return (avg * avg.log()).sum()
