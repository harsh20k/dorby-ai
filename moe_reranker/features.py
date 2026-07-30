"""Bottom-network features for the MoE re-ranker.

This is the "bottom network is more than a fully connected network" slide, made
concrete for this dataset. The mapping:

===================  =====================================================
Slide's block        Here
===================  =====================================================
User feature         seeker profile stats (length, section count)
Item feature         candidate profile stats
Statistic feature    nano cosine, TF-IDF cosine, token overlap, and the
                     *rank* of this candidate among the corpus for this
                     seeker on each channel (the stage-1 retrieval signal)
Context feature      geo overlap, stage gap, role overlap — the cheap
                     engineered versions of the original five failure axes
===================  =====================================================

**Why these and not raw embeddings.** The training set is 111 real pairs. A
shared bottom consuming two 1024-d embeddings would put thousands of parameters
against 111 examples; the slide's architecture assumes a data regime this
project does not have. So the default feature set is ~14 scalars, giving a model
of a few hundred parameters. ``MoEConfig.emb_pca_dims`` can add PCA-reduced
embedding dimensions on top, fitted on train only, for experimenting with that
trade — it is off by default.

Every fitted quantity (standardization, TF-IDF vocabulary, PCA basis) is fit on
the **train split only** and applied to train-dev/holdout, so nothing about the
evaluation population reaches the model. ``FeatureBuilder.fit`` asserts it is
never handed holdout rows.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from baselines.bert_frozen.text import candidate_to_text, seeker_to_text

# Coarse stage vocabulary, ordered. `stage_gap` is the absolute distance between
# the highest stage word found on each side, which is a proxy for the original
# `wrong_stage` axis without needing a label for it.
STAGE_WORDS: list[tuple[str, int]] = [
    ("pre-seed", 0), ("preseed", 0), ("idea stage", 0),
    ("seed", 1),
    ("series a", 2),
    ("series b", 3),
    ("series c", 4), ("series d", 4),
    ("growth", 5), ("late stage", 5), ("pre-ipo", 5), ("public", 6),
]

# Proxy for `wrong_role`: which side of a transaction the text is describing.
ROLE_WORDS = {
    "investor": ("investor", "vc", "venture", "lp", "angel", "fund"),
    "founder": ("founder", "co-founder", "cofounder", "ceo"),
    "operator": ("engineer", "designer", "marketer", "operator", "recruiter", "head of"),
    "customer": ("customer", "buyer", "procurement", "pilot", "design partner"),
}

FEATURE_NAMES: list[str] = [
    "nano_cos",
    "nano_rank_pct",
    "tfidf_cos",
    "tfidf_rank_pct",
    "token_jaccard",
    "seeker_len",
    "cand_len",
    "n_sections",
    "geo_overlap",
    "stage_gap",
    "role_overlap",
    "role_complementary",
    "len_ratio",
    "cos_minus_tfidf",
]


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2}


def _n_sections(looking_for: str | None) -> int:
    if not looking_for:
        return 0
    return len([s for s in re.split(r"\n\s*\n", looking_for.strip()) if s.strip()])


def _stage_level(text: str) -> int | None:
    low = text.lower()
    best = None
    for word, level in STAGE_WORDS:
        if word in low and (best is None or level > best):
            best = level
    return best


def _roles(text: str) -> set[str]:
    low = text.lower()
    return {role for role, words in ROLE_WORDS.items() if any(w in low for w in words)}


def _geo_overlap(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Jaccard over location tokens. Proxy for the `geo_mismatch` axis."""
    ta = _tokens(str(a.get("locationAvailability") or ""))
    tb = _tokens(str(b.get("locationAvailability") or ""))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


@dataclass
class FeatureBuilder:
    """Fits standardization (and optionally TF-IDF/PCA) on train, applies to all."""

    emb_pca_dims: int = 0
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    _mean: np.ndarray | None = None
    _std: np.ndarray | None = None
    _pca_mean: np.ndarray | None = None
    _pca_basis: np.ndarray | None = None
    _fitted: bool = False

    # ------------------------------------------------------------------ raw
    def raw(
        self,
        rows: Sequence[dict[str, Any]],
        *,
        nano_cos: np.ndarray,
        nano_rank_pct: np.ndarray,
        tfidf_cos: np.ndarray,
        tfidf_rank_pct: np.ndarray,
    ) -> np.ndarray:
        """Build the unstandardized feature matrix. No fitted state used."""
        out = np.zeros((len(rows), len(FEATURE_NAMES)), dtype=np.float64)
        for i, r in enumerate(rows):
            seeker = r["userContactFile"]
            cand = r["matchContactFile"]
            s_text = seeker_to_text(seeker, r.get("searchQuery"))
            c_text = candidate_to_text(cand)
            s_tok, c_tok = _tokens(s_text), _tokens(c_text)

            jac = len(s_tok & c_tok) / len(s_tok | c_tok) if (s_tok or c_tok) else 0.0
            s_len, c_len = len(s_tok), len(c_tok)

            s_stage = _stage_level(s_text)
            c_stage = _stage_level(c_text)
            stage_gap = (
                abs(s_stage - c_stage) if (s_stage is not None and c_stage is not None) else 0.0
            )

            s_roles, c_roles = _roles(s_text), _roles(c_text)
            role_overlap = (
                len(s_roles & c_roles) / len(s_roles | c_roles) if (s_roles | c_roles) else 0.0
            )
            # An intro is often good precisely because the two sides are
            # *different* roles (founder <-> investor), so complementarity is a
            # separate signal from overlap rather than its inverse.
            role_complementary = float(
                bool(s_roles and c_roles and not (s_roles & c_roles))
            )

            out[i] = [
                nano_cos[i],
                nano_rank_pct[i],
                tfidf_cos[i],
                tfidf_rank_pct[i],
                jac,
                np.log1p(s_len),
                np.log1p(c_len),
                _n_sections(seeker.get("lookingFor")),
                _geo_overlap(seeker, cand),
                stage_gap,
                role_overlap,
                role_complementary,
                np.log1p(s_len) - np.log1p(c_len),
                nano_cos[i] - tfidf_cos[i],
            ]
        return out

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        raw: np.ndarray,
        *,
        seeker_emb: np.ndarray | None = None,
        cand_emb: np.ndarray | None = None,
        is_holdout: bool = False,
    ) -> "FeatureBuilder":
        if is_holdout:
            raise ValueError(
                "FeatureBuilder.fit was handed holdout rows — fit on train only"
            )
        self._mean = raw.mean(axis=0)
        self._std = raw.std(axis=0)
        # A constant column has zero variance; leave it alone rather than
        # dividing by ~0 and manufacturing a huge feature.
        self._std[self._std < 1e-8] = 1.0

        if self.emb_pca_dims > 0:
            if seeker_emb is None or cand_emb is None:
                raise ValueError("emb_pca_dims > 0 requires seeker_emb and cand_emb")
            stacked = np.concatenate([seeker_emb, cand_emb], axis=1)
            self._pca_mean = stacked.mean(axis=0)
            centered = stacked - self._pca_mean
            # Economy SVD; rows (111) << cols (2048), so at most n_rows
            # components exist and asking for more is meaningless.
            k = min(self.emb_pca_dims, centered.shape[0] - 1, centered.shape[1])
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            self._pca_basis = vt[:k]
            self.feature_names = list(FEATURE_NAMES) + [f"pca_{i}" for i in range(k)]
        else:
            self.feature_names = list(FEATURE_NAMES)

        self._fitted = True
        return self

    # ------------------------------------------------------------ transform
    def transform(
        self,
        raw: np.ndarray,
        *,
        seeker_emb: np.ndarray | None = None,
        cand_emb: np.ndarray | None = None,
    ) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("call fit() before transform()")
        assert self._mean is not None and self._std is not None
        z = (raw - self._mean) / self._std

        if self._pca_basis is not None:
            if seeker_emb is None or cand_emb is None:
                raise ValueError("this builder was fit with PCA; embeddings required")
            assert self._pca_mean is not None
            stacked = np.concatenate([seeker_emb, cand_emb], axis=1) - self._pca_mean
            z = np.concatenate([z, stacked @ self._pca_basis.T], axis=1)

        return z.astype(np.float32)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)
