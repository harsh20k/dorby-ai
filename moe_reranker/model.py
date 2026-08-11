"""Multi-gate mixture-of-experts re-ranker, plus its loss terms.

Architecture, following the MMoE slides:

    features -> shared bottom -> [expert_1 ... expert_M]
                                       |
                     per-task gate g_t(x) = softmax(a_t(x) / tau)
                                       |
                     mixed_t = sum_m g_t,m(x) * expert_m(x)
                                       |
                              per-task head -> logit_t

The "multi-gate" part is **one gate per task**, which is the whole point of the
architecture: related tasks share experts but each learns its own mixing
weights. Here task 0 is the real objective (human accept/decline, 111 training
pairs) and task 1 is an auxiliary target (the LLM judge's opinion, available for
every pair for free). A scarce main task beside an abundant related one is the
regime where MMoE actually earns its keep — with a single task it degenerates
into ordinary MoE.

**Sizing is set by the data, not by taste.** 111 training pairs. Three experts of
4 hidden units over ~14 features is a few hundred parameters. Scaling any of
these up is the fastest way to make the diagnostics meaningless.

Two gate-regularization terms are provided and **both are needed**; they pull in
opposite directions on purpose:

* ``sharpen_loss`` minimizes per-example gate entropy, so each pair commits to
  an expert instead of averaging them (averaging is what lost as
  ``structured_cot`` — see ``docs/llm-judge-experiment.md``).
* ``balance_loss`` maximizes the entropy of the *batch-average* gate, so experts
  stay in use across the dataset. Sharpening alone happily collapses every
  example onto the same expert, which is exactly what Diagnostic 1 detects. The
  slides give the sharpening term and the detector but not this counterweight.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8


class MMoE(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_experts: int = 3,
        expert_hidden: int = 4,
        n_tasks: int = 2,
        tau: float = 0.05,
        expert_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if not 0.0 <= expert_dropout < 1.0:
            raise ValueError("expert_dropout must be in [0, 1)")
        if tau <= 0:
            raise ValueError("tau must be > 0")

        self.n_experts = n_experts
        self.n_tasks = n_tasks
        self.tau = tau
        self.expert_dropout = expert_dropout

        # Experts all see the same input; nothing is pre-assigned to any of them.
        # Specialization, if it happens, is emergent — which is why the
        # diagnostics exist.
        self.experts = nn.ModuleList(
            nn.Sequential(nn.Linear(n_features, expert_hidden), nn.ReLU())
            for _ in range(n_experts)
        )
        # One gate per task.
        self.gates = nn.ModuleList(
            nn.Linear(n_features, n_experts) for _ in range(n_tasks)
        )
        self.heads = nn.ModuleList(nn.Linear(expert_hidden, 1) for _ in range(n_tasks))

    # ------------------------------------------------------------------ core
    def gate_weights(self, x: torch.Tensor) -> torch.Tensor:
        """(B, n_tasks, n_experts) temperature-sharpened routing weights."""
        logits = torch.stack([g(x) for g in self.gates], dim=1)
        if self.training and self.expert_dropout > 0:
            # Drop experts by masking gate logits, so the surviving weights
            # still form a proper distribution after the softmax.
            keep = (
                torch.rand(
                    logits.shape[0], 1, self.n_experts, device=logits.device
                )
                >= self.expert_dropout
            )
            # Never drop every expert for a row.
            all_dropped = ~keep.any(dim=-1, keepdim=True)
            keep = keep | all_dropped
            logits = logits.masked_fill(~keep, float("-inf"))
        return F.softmax(logits / self.tau, dim=-1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits (B, n_tasks), gate weights (B, n_tasks, n_experts))."""
        expert_out = torch.stack([e(x) for e in self.experts], dim=1)  # (B, M, H)
        gates = self.gate_weights(x)  # (B, T, M)
        # (B, T, M, 1) * (B, 1, M, H) -> sum over experts -> (B, T, H)
        mixed = (gates.unsqueeze(-1) * expert_out.unsqueeze(1)).sum(dim=2)
        logits = torch.cat(
            [head(mixed[:, t]) for t, head in enumerate(self.heads)], dim=1
        )
        return logits, gates

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Main-task probability. Task 0 is the real objective by convention."""
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x)
        return torch.sigmoid(logits[:, 0])


# --------------------------------------------------------------------- losses
def sharpen_loss(gates: torch.Tensor) -> torch.Tensor:
    """Mean per-example gate entropy. **Minimize** to polarize routing.

    ``gates``: (B, n_tasks, n_experts). This is the slides' ``L_sharp``.
    """
    ent = -(gates * torch.log(gates + EPS)).sum(dim=-1)
    return ent.mean()


def balance_loss(gates: torch.Tensor) -> torch.Tensor:
    """Negative entropy of the batch-average gate. **Minimize** to keep experts in use.

    Without this counterweight, `sharpen_loss` alone is minimized by sending
    every example to one expert — a collapsed gate that looks confident and has
    learned nothing. Returns a value in [-log M, 0]; minimized (most negative)
    when experts are used equally.
    """
    mean_gate = gates.mean(dim=0)  # (n_tasks, n_experts)
    ent = -(mean_gate * torch.log(mean_gate + EPS)).sum(dim=-1)
    return -ent.mean()


def task_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    weights: tuple[float, ...],
) -> torch.Tensor:
    """Masked, weighted BCE over tasks.

    ``mask`` marks which (row, task) targets exist, so a pair missing an
    auxiliary label contributes nothing to that task instead of being treated as
    a zero.
    """
    total = logits.new_zeros(())
    for t, w in enumerate(weights):
        m = mask[:, t]
        if not bool(m.any()):
            continue
        total = total + w * F.binary_cross_entropy_with_logits(
            logits[m, t], targets[m, t]
        )
    return total
