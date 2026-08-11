"""The two factorizations: label-free SVD of the text matrix, and a low-rank
bilinear correction learned on top of a frozen encoder.

Why the bilinear head is shaped the way it is
---------------------------------------------
Cosine similarity is the special case ``s^T W c`` with ``W = I``: it can only
reward a candidate for pointing the *same way* as the seeker. The real signal
here is not topical sameness — production already filtered on that, which is why
every model in this repo sits at 0.55-0.64 AUC — but complementarity, which is a
statement about *different* directions being compatible. That needs an
off-identity ``W``.

A full ``W`` is ``d x d`` (1M parameters at d=1024) fit on 131 labeled pairs, so
it is parameterized as a rank-``k`` residual instead:

    score(s, c) = s.c + (A s).(B c) = s^T (I + A^T B) c,   A, B in R^{k x d}

At ``init_scale -> 0`` the head *is* frozen cosine, so the experiment reads as a
clean "did learning a correction help, starting from the baseline". Parameter
count is ``2kd`` (4k at k=16, d=128), which is still large against 131 examples —
hence the weight decay, the inner-CV hyperparameter search, and the
label-permutation null in ``evaluate.py``. Any gain smaller than that null is
not a result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def l2_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.clip(norms, eps, None)


# ---------------------------------------------------------------------------
# Arm 1: classic text matrix factorization (LSA / truncated SVD)
# ---------------------------------------------------------------------------


@dataclass
class SvdReducer:
    """Truncated SVD of the stacked text matrix, refit-free at transform time.

    Fit is **label-free**, so fitting it on every text in the population is not
    leakage in the label sense — it is the same transductive assumption the
    published TF-IDF baseline already makes by fitting its vocabulary/IDF on the
    full corpus. Kept identical across both arms so they stay comparable.
    """

    n_components: int
    seed: int = 0
    _components: np.ndarray | None = None  # (k, d)
    _mean: np.ndarray | None = None

    def fit(self, matrix: np.ndarray, *, center: bool = False) -> "SvdReducer":
        from sklearn.decomposition import TruncatedSVD

        x = np.asarray(matrix, dtype=np.float64)
        # LSA convention is *no* centering (keeps the sparse structure
        # meaningful); centering is offered for the dense Voyage backbone where
        # the shared mean direction otherwise dominates component 1.
        self._mean = x.mean(axis=0) if center else np.zeros(x.shape[1])
        k = min(self.n_components, min(x.shape) - 1)
        svd = TruncatedSVD(n_components=k, random_state=self.seed)
        svd.fit(x - self._mean)
        self._components = svd.components_.astype(np.float32)
        self.explained_variance_ratio_ = float(svd.explained_variance_ratio_.sum())
        return self

    def transform(self, matrix: np.ndarray, *, normalize: bool = True) -> np.ndarray:
        if self._components is None:
            raise RuntimeError("call fit() first")
        out = (np.asarray(matrix, dtype=np.float32) - self._mean) @ self._components.T
        return l2_normalize(out.astype(np.float32)) if normalize else out.astype(np.float32)


# ---------------------------------------------------------------------------
# Arm 2: low-rank bilinear scorer
# ---------------------------------------------------------------------------


@dataclass
class BilinearScorer:
    """``score(s, c) = s.c + (A s).(B c)``, trained with logistic loss."""

    A: np.ndarray  # (k, d)
    B: np.ndarray  # (k, d)
    scale: float
    bias: float
    rank: int
    dim: int
    train_loss: list[float]

    def pair_scores(self, seeker: np.ndarray, cand: np.ndarray) -> np.ndarray:
        """Row-wise scores for aligned (seeker, candidate) rows."""
        base = np.sum(seeker * cand, axis=-1)
        residual = np.sum((seeker @ self.A.T) * (cand @ self.B.T), axis=-1)
        return (base + residual).astype(np.float32)

    def score_matrix(self, queries: np.ndarray, corpus: np.ndarray) -> np.ndarray:
        """(n_queries x n_candidates) scores, for the retrieval metrics."""
        base = queries @ corpus.T
        residual = (queries @ self.A.T) @ (corpus @ self.B.T).T
        return (base + residual).astype(np.float32)

    def to_json(self) -> dict:
        w_norm = float(np.linalg.norm(self.A.T @ self.B))
        return {
            "rank": self.rank,
            "dim": self.dim,
            "scale": self.scale,
            "bias": self.bias,
            "residual_frobenius_norm": w_norm,
            "final_loss": self.train_loss[-1] if self.train_loss else None,
        }


def train_bilinear(
    seeker: np.ndarray,
    cand: np.ndarray,
    labels: np.ndarray,
    *,
    rank: int,
    weight_decay: float,
    lr: float,
    steps: int,
    init_scale: float,
    seed: int,
) -> BilinearScorer:
    """Full-batch Adam on the logistic loss. Tiny model, tiny data, no minibatching.

    ``scale``/``bias`` are learned alongside so the logistic loss has a
    calibrated operating point; neither affects AUC or ranking (both are
    monotone transforms), so they cannot manufacture a metric gain.
    """
    import torch

    torch.manual_seed(seed)
    d = seeker.shape[1]
    S = torch.tensor(np.asarray(seeker, dtype=np.float32))
    C = torch.tensor(np.asarray(cand, dtype=np.float32))
    y = torch.tensor(np.asarray(labels, dtype=np.float32))

    g = torch.Generator().manual_seed(seed)
    A = torch.nn.Parameter(torch.randn(rank, d, generator=g) * init_scale)
    B = torch.nn.Parameter(torch.randn(rank, d, generator=g) * init_scale)
    scale = torch.nn.Parameter(torch.tensor(10.0))
    bias = torch.nn.Parameter(torch.tensor(0.0))

    opt = torch.optim.Adam([A, B, scale, bias], lr=lr)
    bce = torch.nn.BCEWithLogitsLoss()
    losses: list[float] = []

    for _ in range(steps):
        opt.zero_grad()
        base = (S * C).sum(-1)
        residual = ((S @ A.T) * (C @ B.T)).sum(-1)
        logits = scale * (base + residual) + bias
        # Decay applied only to the residual, never to the cosine term — the
        # baseline must remain the zero-penalty solution.
        loss = bce(logits, y) + weight_decay * (A.pow(2).sum() + B.pow(2).sum())
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))

    return BilinearScorer(
        A=A.detach().numpy(),
        B=B.detach().numpy(),
        scale=float(scale.detach()),
        bias=float(bias.detach()),
        rank=rank,
        dim=d,
        train_loss=losses,
    )


def cosine_pair_scores(seeker: np.ndarray, cand: np.ndarray) -> np.ndarray:
    return np.sum(seeker * cand, axis=-1).astype(np.float32)
