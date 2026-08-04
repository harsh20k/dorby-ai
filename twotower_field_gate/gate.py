"""A small learned combiner over N piece embeddings.

Input-dependent attention gate: takes the stacked piece vectors for one
seeker, computes one gate logit per piece from a tiny linear layer over the
*concatenation* of all pieces (so the gate can see all pieces at once,
not just each one in isolation), softmaxes, and returns the weighted sum.
This is the learned generalization of the fixed `alpha` blend used in
`query_weighted/` — instead of one hand-picked weight for the whole project,
each seeker gets its own combination, learned from data.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FieldGate(nn.Module):
    def __init__(self, n_pieces: int, dim: int) -> None:
        super().__init__()
        self.n_pieces = n_pieces
        self.gate = nn.Linear(n_pieces * dim, n_pieces)
        # Start near-uniform: zero-init keeps the gate at softmax([0,0,0]) = uniform
        # weights at step 0, so training starts from "simple average," not noise.
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, piece_embs: list[torch.Tensor]) -> torch.Tensor:
        """piece_embs: list of (B, dim) tensors, one per piece (same order every call)."""
        stacked = torch.stack(piece_embs, dim=1)  # (B, n_pieces, dim)
        flat = stacked.reshape(stacked.size(0), -1)  # (B, n_pieces*dim)
        logits = self.gate(flat)  # (B, n_pieces)
        weights = F.softmax(logits, dim=-1)  # (B, n_pieces)
        combined = (stacked * weights.unsqueeze(-1)).sum(dim=1)  # (B, dim)
        return F.normalize(combined, dim=-1)
