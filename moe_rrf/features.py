"""Text-only bottom-network features, computable on real *and* synthetic pairs.

**Why this is a copy and not a flag on ``moe_reranker/features.py``.** Per the
experiment-isolation rule in CLAUDE.md, a new experiment gets a new package; the
previous experiment's files stay byte-identical so its published numbers remain
reproducible. Only the parts that genuinely differ are duplicated —
``moe_reranker.model`` and ``moe_reranker.diagnostics`` need no change and are
imported unchanged rather than copied.

**What differs from ``moe_reranker/features.py``.** That version's two strongest
features are ``nano_cos`` and ``nano_rank_pct``, which need cached
``voyage-4-nano`` embeddings. Those exist for the 200 real seed pairs but not for
the 2,619 synthetic pairs, whose cached vectors are Qwen3-Embedding-8B — a
different space. Mixing spaces across the two populations would make the feature
mean different things per row, so the embedding channel is **dropped entirely**
and everything here is derived from text.

The cost is real and should be stated: the strongest single dense signal is gone.
The mitigation is that TF-IDF was already carrying most of the weight anyway —
`docs/baseline-results-holdout.md`'s hybrid fits alpha ≈ 0.95 onto the lexical
channel — and dropping it is the only option that keeps one feature definition
valid across both populations without a multi-hour encode or a Modal GPU run.

One feature is *added* relative to the earlier set: ``query_cand_jaccard``, the
searchQuery-to-candidate lexical overlap. The query ablation
(`docs/baseline-results-all.md`) showed `searchQuery` is load-bearing rather than
redundant with the profile, and unlike the nano channel it costs nothing here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from baselines.bert_frozen.text import candidate_to_text, seeker_to_text

# Reused unchanged from the earlier experiment: same axis proxies, same word
# lists, so the geo/stage/role features mean the same thing in both packages.
from moe_reranker.features import (
    ROLE_WORDS,
    STAGE_WORDS,
    _geo_overlap,
    _n_sections,
    _roles,
    _stage_level,
    _tokens,
)

FEATURE_NAMES: list[str] = [
    "tfidf_cos",
    "tfidf_rank_pct",
    "token_jaccard",
    "query_cand_jaccard",
    "seeker_len",
    "cand_len",
    "len_ratio",
    "n_sections",
    "geo_overlap",
    "stage_gap",
    "role_overlap",
    "role_complementary",
]


def build_raw(
    rows: Sequence[dict[str, Any]],
    *,
    tfidf_cos: np.ndarray,
    tfidf_rank_pct: np.ndarray,
) -> np.ndarray:
    """Unstandardized feature matrix. Uses no fitted state, so it is population-agnostic."""
    out = np.zeros((len(rows), len(FEATURE_NAMES)), dtype=np.float64)
    for i, r in enumerate(rows):
        seeker = r["userContactFile"] or {}
        cand = r["matchContactFile"] or {}
        query = r.get("searchQuery") or ""

        s_text = seeker_to_text(seeker, query)
        c_text = candidate_to_text(cand)
        s_tok, c_tok, q_tok = _tokens(s_text), _tokens(c_text), _tokens(query)

        jac = len(s_tok & c_tok) / len(s_tok | c_tok) if (s_tok or c_tok) else 0.0
        q_jac = len(q_tok & c_tok) / len(q_tok | c_tok) if (q_tok or c_tok) else 0.0
        s_len, c_len = len(s_tok), len(c_tok)

        s_stage, c_stage = _stage_level(s_text), _stage_level(c_text)
        stage_gap = (
            abs(s_stage - c_stage)
            if (s_stage is not None and c_stage is not None)
            else 0.0
        )

        s_roles, c_roles = _roles(s_text), _roles(c_text)
        role_overlap = (
            len(s_roles & c_roles) / len(s_roles | c_roles) if (s_roles | c_roles) else 0.0
        )
        role_complementary = float(bool(s_roles and c_roles and not (s_roles & c_roles)))

        out[i] = [
            tfidf_cos[i],
            tfidf_rank_pct[i],
            jac,
            q_jac,
            np.log1p(s_len),
            np.log1p(c_len),
            np.log1p(s_len) - np.log1p(c_len),
            _n_sections(seeker.get("lookingFor")),
            _geo_overlap(seeker, cand),
            stage_gap,
            role_overlap,
            role_complementary,
        ]
    return out


def tfidf_channel(
    rows: Sequence[dict[str, Any]], fit_rows: Sequence[dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray]:
    """TF-IDF cosine + within-population rank percentile.

    ``fit_rows`` decides the vocabulary/IDF. **The rule used throughout this
    experiment: fit on whatever population the model trains on**, never on the
    population it is evaluated against. For the synth-trained arms that means the
    vectorizer sees synthetic text only and the real pairs are transformed by it
    cold — which is the honest setup, and part of what is being measured
    (vocabulary transfer is one way synthetic-to-real can fail).

    ``tfidf_rank_pct`` is this candidate's percentile among all candidates
    *within the same population*, standing in for the stage-1 retrieval rank.

    Uses ``baselines.tfidf.encode.TfidfEncoder`` unchanged rather than a local
    ``TfidfVectorizer``. That is deliberate: it is the repo's single source of
    truth for the lexical channel, so these numbers stay comparable to
    ``docs/baseline-results-holdout.md``. A first version of this function rolled
    its own vectorizer with ``ngram_range=(1,1)`` and ``sublinear_tf=True``,
    against the encoder's ``(1,2)`` and no sublinear scaling; on the 131 real
    pairs that scored **0.4366 AUC versus the encoder's 0.5660** (correlation only
    0.52), i.e. the reimplementation was actively broken and would have made every
    downstream feature untrustworthy. Reuse beats re-derivation here.

    **``TfidfEncoder.encode()`` must be bypassed here, and the reason is a real
    bug it would otherwise cause.** Its disk cache keys on
    ``(texts, max_features, ngram_range)`` — *not* on the fitted vocabulary. This
    function deliberately encodes the **same** rows under **different** fits (real
    vocabulary vs synthetic vocabulary), which collide on that key, so the second
    call silently returns the first call's vectors. The symptom was unmistakable
    once looked for: the real-vocabulary and synthetic-vocabulary arms reported
    byte-identical AUCs of 0.5660. CLAUDE.md flags this cache hazard for
    ``cache_name``; it applies to the content-hashed default too. So we fit with
    ``TfidfEncoder`` (to inherit its exact parameters) and call
    ``vectorizer.transform`` directly, which touches no cache.
    """
    from baselines.tfidf.encode import TfidfEncoder

    def texts(rs: Sequence[dict[str, Any]]) -> tuple[list[str], list[str]]:
        return (
            [
                seeker_to_text(r["userContactFile"] or {}, r.get("searchQuery") or "")
                for r in rs
            ],
            [candidate_to_text(r["matchContactFile"] or {}) for r in rs],
        )

    fit_s, fit_c = texts(fit_rows)
    enc = TfidfEncoder()
    enc.fit(fit_s + fit_c)
    assert enc.vectorizer is not None

    s_texts, c_texts = texts(rows)
    S = enc.vectorizer.transform(s_texts).toarray().astype(np.float32)
    C = enc.vectorizer.transform(c_texts).toarray().astype(np.float32)

    # TfidfEncoder emits L2-normalized rows, so a dot product is already a cosine.
    cos = (S * C).sum(axis=1)
    full = S @ C.T
    rank_pct = np.array(
        [(full[i] < full[i, i]).mean() for i in range(full.shape[0])], dtype=np.float64
    )
    return cos, rank_pct


@dataclass
class Standardizer:
    """Z-scoring fitted on the training population only."""

    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))

    def fit(self, raw: np.ndarray) -> "Standardizer":
        self.mean = raw.mean(axis=0)
        self.std = raw.std(axis=0)
        # A constant column has no information; leave it rather than dividing by ~0.
        self.std[self.std < 1e-8] = 1.0
        return self

    def transform(self, raw: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("call fit() before transform()")
        return ((raw - self.mean) / self.std).astype(np.float32)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)
