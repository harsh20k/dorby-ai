"""Build the per-(pair, section) input rows.

Each row answers a narrower question than the pair-level model does: *does this
candidate satisfy **this specific ask**?* Three blocks go in.

===============  ====  ===================================================
block            dims  what it is
===============  ====  ===================================================
similarity          3  section-to-candidate cosine, rank percentile among
                       all candidates, and cosine minus the pair-level
                       lexical cosine (where semantics and keywords disagree)
interaction        32  elementwise product of the section and candidate
                       vectors, projected by a learned layer inside the model
pair scalars       12  the existing feature table, repeated on every row of
                       a pair, imported unchanged from ``moe_rrf.features``
===============  ====  ===================================================

The interaction block is deliberately *not* projected here: the projection is a
learned layer in ``model.py``, so it is fit with the rest of the network rather
than by an unsupervised step that cannot know what matters. What this module
emits is the raw elementwise product; the model owns the reduction.

Reused read-only from earlier experiments: ``moe_rrf.features.build_raw`` and
``tfidf_channel`` for the 12 pair scalars. Copying them would have duplicated a
function whose numbers are already pinned by tests, and importing is what the
isolation rule asks for when nothing needs to change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from baselines.bert_frozen.text import candidate_to_text

from .encode import Encoder
from .sections import Section, sections_for_pair

#: Names for the 3 similarity scalars; the other blocks are unnamed vectors.
SIM_FEATURE_NAMES: list[str] = ["sec_cos", "sec_rank_pct", "sec_cos_minus_pair"]


@dataclass
class SectionRow:
    """One training row: a pair, one of its asks, and the label of the pair.

    ``pair_index`` is what lets pooling gather a pair's rows back together, and
    ``seeker_id`` is what keeps whole seekers inside one CV fold. Neither is ever
    a feature — seeker identity alone predicted the label at 0.687 AUC on an
    earlier synthetic batch, so it must shape the split, never the input.
    """

    pair_index: int
    seeker_id: str
    section: Section
    label: float


@dataclass
class SectionFeatures:
    """The assembled matrices for one population."""

    rows: list[SectionRow]
    #: (N, 3) similarity scalars.
    sim: np.ndarray
    #: (N, D) elementwise product of section and candidate embeddings.
    interaction: np.ndarray
    #: (N, D) section embedding — the gate's only input.
    section_emb: np.ndarray
    #: (N, 12) pair-level scalars, repeated across a pair's rows.
    pair_scalars: np.ndarray
    #: (P,) pair-level labels, aligned with ``pair_index``.
    pair_labels: np.ndarray
    #: (P,) seeker per pair.
    pair_seekers: list[str]

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_pairs(self) -> int:
        return len(self.pair_labels)

    def groups(self) -> list[np.ndarray]:
        """Row indices belonging to each pair, in pair order."""
        out: list[list[int]] = [[] for _ in range(self.n_pairs)]
        for i, r in enumerate(self.rows):
            out[r.pair_index].append(i)
        return [np.array(g, dtype=np.int64) for g in out]


def build_section_features(
    pairs: Sequence[dict[str, Any]],
    labels: Sequence[float],
    seeker_ids: Sequence[str],
    *,
    encoder: Encoder,
    pair_scalars: np.ndarray,
    pair_tfidf_cos: np.ndarray,
    max_sections: int = 8,
    min_section_chars: int = 40,
) -> SectionFeatures:
    """Explode pairs into section rows and compute every block.

    ``pair_scalars`` and ``pair_tfidf_cos`` come from ``moe_rrf.features`` and
    are indexed by pair; they are broadcast onto each of that pair's rows.
    """
    if len(pairs) != len(labels) != len(seeker_ids):
        raise ValueError("pairs, labels and seeker_ids must be the same length")

    rows: list[SectionRow] = []
    sec_texts: list[str] = []
    cand_texts: list[str] = []
    kept_pair_indices: list[int] = []

    for p_i, pair in enumerate(pairs):
        secs = sections_for_pair(
            pair, min_chars=min_section_chars, max_sections=max_sections
        )
        if not secs:
            continue
        kept_pair_indices.append(p_i)
        c_text = candidate_to_text(pair.get("matchContactFile") or {})
        for s in secs:
            rows.append(
                SectionRow(
                    pair_index=len(kept_pair_indices) - 1,
                    seeker_id=seeker_ids[p_i],
                    section=s,
                    label=float(labels[p_i]),
                )
            )
            sec_texts.append(s.text)
            cand_texts.append(c_text)

    if not rows:
        raise ValueError("no sections produced from any pair")

    S = encoder.encode(sec_texts, role="query")
    C = encoder.encode(cand_texts, role="document")

    # Rows are L2-normalized, so the row-wise dot product is already a cosine.
    cos = (S * C).sum(axis=1)

    # Rank percentile: where this candidate sits among *all* candidates in the
    # population for this same ask. Stands in for the stage-1 retrieval rank,
    # which is what a re-ranker would actually receive at serving time.
    uniq_cand, inv = np.unique(np.array(cand_texts), return_inverse=True)
    Cu = encoder.encode(list(uniq_cand), role="document")
    all_scores = S @ Cu.T  # (N_rows, N_unique_candidates)
    rank_pct = (all_scores < cos[:, None]).mean(axis=1)

    keep = np.array(kept_pair_indices, dtype=np.int64)
    row_pair = np.array([r.pair_index for r in rows], dtype=np.int64)
    pair_cos_per_row = pair_tfidf_cos[keep][row_pair]

    sim = np.stack([cos, rank_pct, cos - pair_cos_per_row], axis=1).astype(np.float32)

    return SectionFeatures(
        rows=rows,
        sim=sim,
        interaction=(S * C).astype(np.float32),
        section_emb=S.astype(np.float32),
        pair_scalars=pair_scalars[keep][row_pair].astype(np.float32),
        pair_labels=np.array([labels[i] for i in keep], dtype=np.float32),
        pair_seekers=[seeker_ids[i] for i in keep],
    )


@dataclass
class EmbeddingReducer:
    """PCA on the embedding blocks, fitted on training rows only.

    **Why this is mandatory rather than optional.** The interaction and gate
    blocks feed learned ``Linear`` layers whose parameter count is the embedding
    width times the output width. At full width those two layers hold ~960k
    parameters for TF-IDF (20,000-d) or ~197k for Qwen3 (4,096-d), against 708
    training rows — so the projections *are* the model, and they will fit noise.
    Runs ``sec_001`` and ``sec_002`` were done without this and their arm
    rankings reversed completely between the two encoders, which is the
    signature of exactly that.

    One basis is fitted on the concatenation of the section and interaction
    blocks so both land in the same reduced space, and it is fitted on training
    rows only — a basis fitted on the evaluation rows would leak their structure
    into training.
    """

    n_components: int = 48
    mean: np.ndarray | None = None
    basis: np.ndarray | None = None

    def fit(self, section_emb: np.ndarray, interaction: np.ndarray) -> "EmbeddingReducer":
        x = np.concatenate([section_emb, interaction], axis=0).astype(np.float64)
        self.mean = x.mean(axis=0)
        centered = x - self.mean
        # Economy SVD: n_rows is far smaller than n_features here, so this is
        # cheap even at 20,000 dims.
        k = min(self.n_components, min(centered.shape) - 1)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        self.basis = vt[:k].T.astype(np.float32)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.basis is None:
            raise RuntimeError("call fit() before transform()")
        return ((x - self.mean) @ self.basis).astype(np.float32)

    @property
    def out_dims(self) -> int:
        if self.basis is None:
            raise RuntimeError("call fit() before out_dims")
        return int(self.basis.shape[1])


@dataclass
class RowStandardizer:
    """Z-score the similarity and pair-scalar blocks on the training rows only.

    The embedding blocks are left alone: they are already L2-normalized, and
    per-dimension standardization of an embedding destroys the geometry that
    makes the dot product meaningful.
    """

    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    names: list[str] = field(default_factory=lambda: list(SIM_FEATURE_NAMES))

    def fit(self, sim: np.ndarray, pair_scalars: np.ndarray) -> "RowStandardizer":
        x = np.concatenate([sim, pair_scalars], axis=1)
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        self.std[self.std < 1e-8] = 1.0
        return self

    def transform(self, sim: np.ndarray, pair_scalars: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("call fit() before transform()")
        x = np.concatenate([sim, pair_scalars], axis=1)
        return ((x - self.mean) / self.std).astype(np.float32)
