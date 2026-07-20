"""Late-fusion helpers for TF-IDF + Voyage-4-nano cosine scores.

Two modes:
  - ``alpha``: score = α · z(tfidf) + (1−α) · z(voyage), α chosen by fit-set AUC.
  - ``logistic``: 2-feature logistic regression on raw [tfidf, voyage] scores.

Z-scoring uses fit-set mean/std per channel so the two score distributions
are on a comparable scale before blending. Holdout scores are transformed
with the same fit-set statistics (no holdout leakage into normalization).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FusionMode = Literal["alpha", "logistic"]


@dataclass(frozen=True)
class ZStats:
    mean: float
    std: float

    def transform(self, x: np.ndarray) -> np.ndarray:
        denom = self.std if self.std > 1e-12 else 1.0
        return (x - self.mean) / denom


@dataclass
class AlphaFusion:
    mode: Literal["alpha"]
    alpha: float
    tfidf_z: ZStats
    voyage_z: ZStats
    fit_auc: float
    alpha_grid: list[float]
    alpha_aucs: list[float]

    def pair_scores(self, tfidf: np.ndarray, voyage: np.ndarray) -> np.ndarray:
        return self.alpha * self.tfidf_z.transform(tfidf) + (
            1.0 - self.alpha
        ) * self.voyage_z.transform(voyage)

    def score_matrix(
        self, tfidf_matrix: np.ndarray, voyage_matrix: np.ndarray
    ) -> np.ndarray:
        return self.pair_scores(tfidf_matrix, voyage_matrix)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "alpha": self.alpha,
            "tfidf_z": asdict(self.tfidf_z),
            "voyage_z": asdict(self.voyage_z),
            "fit_auc": self.fit_auc,
            "alpha_grid": self.alpha_grid,
            "alpha_aucs": self.alpha_aucs,
        }


@dataclass
class LogisticFusion:
    mode: Literal["logistic"]
    # Coefs are on StandardScaler-normalized features (fit-set mean/std).
    coef_tfidf: float
    coef_voyage: float
    intercept: float
    tfidf_mean: float
    tfidf_std: float
    voyage_mean: float
    voyage_std: float
    fit_auc: float
    n_fit: int
    C: float

    def _scale(self, tfidf: np.ndarray, voyage: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t_std = self.tfidf_std if self.tfidf_std > 1e-12 else 1.0
        v_std = self.voyage_std if self.voyage_std > 1e-12 else 1.0
        return (
            (np.asarray(tfidf, dtype=np.float64) - self.tfidf_mean) / t_std,
            (np.asarray(voyage, dtype=np.float64) - self.voyage_mean) / v_std,
        )

    def pair_scores(self, tfidf: np.ndarray, voyage: np.ndarray) -> np.ndarray:
        # Decision function is monotonic with P(pos); use it directly so
        # retrieval ranking doesn't compress near 0/1.
        zt, zv = self._scale(tfidf, voyage)
        return self.intercept + self.coef_tfidf * zt + self.coef_voyage * zv

    def score_matrix(
        self, tfidf_matrix: np.ndarray, voyage_matrix: np.ndarray
    ) -> np.ndarray:
        return self.pair_scores(tfidf_matrix, voyage_matrix)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "coef_tfidf": self.coef_tfidf,
            "coef_voyage": self.coef_voyage,
            "intercept": self.intercept,
            "tfidf_mean": self.tfidf_mean,
            "tfidf_std": self.tfidf_std,
            "voyage_mean": self.voyage_mean,
            "voyage_std": self.voyage_std,
            "fit_auc": self.fit_auc,
            "n_fit": self.n_fit,
            "C": self.C,
        }


FusionModel = AlphaFusion | LogisticFusion


def _z_stats(x: np.ndarray) -> ZStats:
    return ZStats(mean=float(np.mean(x)), std=float(np.std(x)))


def _pair_auc(y: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, scores))


def fit_alpha_fusion(
    tfidf_scores: np.ndarray,
    voyage_scores: np.ndarray,
    labels: np.ndarray,
    *,
    grid: np.ndarray | None = None,
) -> AlphaFusion:
    """Choose α maximizing fit-set pair AUC on z-scored channel blend."""
    tfidf = np.asarray(tfidf_scores, dtype=np.float64)
    voyage = np.asarray(voyage_scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int32)
    if len(tfidf) != len(voyage) or len(tfidf) != len(y):
        raise ValueError("tfidf/voyage/labels length mismatch")

    tfidf_z = _z_stats(tfidf)
    voyage_z = _z_stats(voyage)
    zt = tfidf_z.transform(tfidf)
    zv = voyage_z.transform(voyage)

    if grid is None:
        grid = np.linspace(0.0, 1.0, 21)
    grid = np.asarray(grid, dtype=np.float64)

    best_alpha = 0.5
    best_auc = -1.0
    aucs: list[float] = []
    for alpha in grid:
        scores = alpha * zt + (1.0 - alpha) * zv
        auc = _pair_auc(y, scores)
        aucs.append(auc)
        if auc == auc and auc > best_auc:  # skip NaN
            best_auc = auc
            best_alpha = float(alpha)

    return AlphaFusion(
        mode="alpha",
        alpha=best_alpha,
        tfidf_z=tfidf_z,
        voyage_z=voyage_z,
        fit_auc=float(best_auc),
        alpha_grid=[float(a) for a in grid],
        alpha_aucs=aucs,
    )


def fit_logistic_fusion(
    tfidf_scores: np.ndarray,
    voyage_scores: np.ndarray,
    labels: np.ndarray,
    *,
    C: float = 1.0,
    seed: int = 42,
) -> LogisticFusion:
    """Fit a 2-feature logistic regression on z-scored cosine scores."""
    tfidf = np.asarray(tfidf_scores, dtype=np.float64)
    voyage = np.asarray(voyage_scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int32)
    if len(tfidf) != len(voyage) or len(tfidf) != len(y):
        raise ValueError("tfidf/voyage/labels length mismatch")
    if len(np.unique(y)) < 2:
        raise ValueError("need both classes in fit set")

    X = np.column_stack([tfidf, voyage])
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=seed,
                ),
            ),
        ]
    )
    pipe.fit(X, y)
    scale: StandardScaler = pipe.named_steps["scale"]
    clf: LogisticRegression = pipe.named_steps["clf"]
    decision = pipe.decision_function(X)
    return LogisticFusion(
        mode="logistic",
        coef_tfidf=float(clf.coef_[0, 0]),
        coef_voyage=float(clf.coef_[0, 1]),
        intercept=float(clf.intercept_[0]),
        tfidf_mean=float(scale.mean_[0]),
        tfidf_std=float(scale.scale_[0]),
        voyage_mean=float(scale.mean_[1]),
        voyage_std=float(scale.scale_[1]),
        fit_auc=_pair_auc(y, decision),
        n_fit=int(len(y)),
        C=float(C),
    )


def fit_fusion(
    mode: FusionMode,
    tfidf_scores: np.ndarray,
    voyage_scores: np.ndarray,
    labels: np.ndarray,
    *,
    C: float = 1.0,
    seed: int = 42,
) -> FusionModel:
    if mode == "alpha":
        return fit_alpha_fusion(tfidf_scores, voyage_scores, labels)
    if mode == "logistic":
        return fit_logistic_fusion(
            tfidf_scores, voyage_scores, labels, C=C, seed=seed
        )
    raise ValueError(f"unknown fusion mode: {mode}")


def rrf_score_matrix(
    tfidf_matrix: np.ndarray,
    voyage_matrix: np.ndarray,
    *,
    k: int = 60,
) -> np.ndarray:
    """Reciprocal rank fusion of two score matrices (higher = better)."""
    if tfidf_matrix.shape != voyage_matrix.shape:
        raise ValueError("tfidf/voyage matrix shape mismatch")

    def _rrf(mat: np.ndarray) -> np.ndarray:
        # rank 1 = best (highest score); stable sort
        order = np.argsort(-mat, axis=1, kind="stable")
        ranks = np.empty_like(order)
        row_idx = np.arange(mat.shape[0])[:, None]
        ranks[row_idx, order] = np.arange(1, mat.shape[1] + 1)[None, :]
        return 1.0 / (k + ranks.astype(np.float64))

    return _rrf(tfidf_matrix) + _rrf(voyage_matrix)
