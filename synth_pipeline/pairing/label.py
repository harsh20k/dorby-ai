"""Label pairs with the TF-IDF + Voyage-nano late-fusion scorer.

No LLM judge (deliberate — see the package docstring). The fusion scorer is the
strongest pair scorer measured on the matched 69-pair holdout in
`docs/baseline-results-holdout.md`:

    hybrid TF-IDF+nano   pair AUC 0.6397   hard-neg AUC 0.6034
    voyage-4-large       pair AUC 0.6086   hard-neg AUC 0.6017
    tfidf alone          pair AUC 0.5922   hard-neg AUC 0.5017
    voyage-4-nano alone  pair AUC 0.5793   hard-neg AUC 0.5707

It is still weak in absolute terms — accuracy at its own best-F1 threshold is
0.594 — which is why labeling uses a DEADBAND rather than a single cut. Pairs whose
score lands near the boundary are coin flips; giving them a confident label would
manufacture noise at exactly the decision boundary that matters most. Excluding
them costs yield, which at this scale is free.

The fusion is fitted on real TRAIN pairs only (`include_synth=False`), so the frozen
69-pair holdout never touches the fit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from baselines.bert_frozen.text import candidate_to_text, seeker_to_text
from baselines.hybrid_tfidf_voyage.fusion import FusionModel, fit_fusion
from baselines.metrics import _best_f1_threshold
from baselines.tfidf.encode import TfidfEncoder, cosine_scores as tfidf_cosine
from baselines.voyage_nano.encode import VoyageNanoEncoder, cosine_scores as voyage_cosine
from twotower.data import build_split_bundle


@dataclass
class Scorer:
    tfidf: TfidfEncoder
    voyage: VoyageNanoEncoder
    fusion: FusionModel
    fit_scores: np.ndarray
    fit_labels: np.ndarray
    n_fit: int
    meta: dict[str, Any] = field(default_factory=dict)

    def score(self, seeker_texts: list[str], cand_texts: list[str], *, cache_prefix: str,
              batch_size: int = 4) -> np.ndarray:
        t_s = self.tfidf.encode(seeker_texts, cache_name=f"{cache_prefix}_seeker")
        t_c = self.tfidf.encode(cand_texts, cache_name=f"{cache_prefix}_cand")
        v_s = self.voyage.encode(seeker_texts, role="query", batch_size=batch_size,
                                 cache_name=f"{cache_prefix}_seeker")
        v_c = self.voyage.encode(cand_texts, role="document", batch_size=batch_size,
                                 cache_name=f"{cache_prefix}_cand")
        return self.fusion.pair_scores(tfidf_cosine(t_s, t_c), voyage_cosine(v_s, v_c))


def fit_scorer(
    data_dir: Path,
    split_path: Path,
    *,
    cache_dir: Path,
    fusion_mode: str = "alpha",
    voyage_model: str = "voyageai/voyage-4-nano",
    max_length: int = 8192,
    truncate_dim: int | None = 1024,
    batch_size: int = 4,
    seed: int = 42,
) -> Scorer:
    """Fit TF-IDF vocab + fusion weights on real train pairs only."""
    bundle = build_split_bundle(data_dir, split_path, include_synth=False, seed=seed)
    fit_pairs = sorted(bundle.train + bundle.train_dev, key=lambda p: p.pair_id)
    if not fit_pairs:
        raise RuntimeError("empty fit set — check data_dir/split_path")

    seeker_texts = [p.seeker_text for p in fit_pairs]
    cand_texts = [p.candidate_text for p in fit_pairs]
    labels = np.array([p.y for p in fit_pairs], dtype=np.int32)

    tfidf = TfidfEncoder(cache_dir=Path(cache_dir) / "tfidf")
    tfidf.fit(seeker_texts + cand_texts)

    voyage = VoyageNanoEncoder(
        model_name=voyage_model,
        max_length=max_length,
        truncate_dim=truncate_dim,
        cache_dir=Path(cache_dir) / "voyage",
    )

    t_s = tfidf.encode(seeker_texts, cache_name="fit_seeker")
    t_c = tfidf.encode(cand_texts, cache_name="fit_cand")
    v_s = voyage.encode(seeker_texts, role="query", batch_size=batch_size,
                        cache_name="fit_seeker")
    v_c = voyage.encode(cand_texts, role="document", batch_size=batch_size,
                        cache_name="fit_cand")

    tfidf_scores = tfidf_cosine(t_s, t_c)
    voyage_scores = voyage_cosine(v_s, v_c)
    fusion = fit_fusion(fusion_mode, tfidf_scores, voyage_scores, labels, seed=seed)
    fit_scores = fusion.pair_scores(tfidf_scores, voyage_scores)

    return Scorer(
        tfidf=tfidf,
        voyage=voyage,
        fusion=fusion,
        fit_scores=fit_scores,
        fit_labels=labels,
        n_fit=len(fit_pairs),
        meta={
            "fusion": fusion.to_dict(),
            "split_hash": bundle.split_hash,
            "n_fit_pairs": len(fit_pairs),
            "n_fit_pos": int(labels.sum()),
            "n_fit_neg": int(len(labels) - labels.sum()),
        },
    )


@dataclass(frozen=True)
class Thresholds:
    center: float
    lower: float
    upper: float
    margin: float
    fit_std: float
    fit_best_f1: float
    mode: str = "absolute"


def compute_thresholds(scorer: Scorer, *, margin: float = 0.25) -> Thresholds:
    """Deadband around the fit-set best-F1 threshold, in fit-score std units.

    WARNING: measured not to transfer to synthetic pairs. On the pair_test_001
    batch the synthetic scores ran 0.57..9.86 against a real-pair threshold of
    about -2.18 — the two distributions do not overlap at all, so every pair
    labeled positive. Two compounding causes: synthetic profiles are far more
    homogeneous than real contacts (one model, one style spec, ~8 archetypes),
    and `select.py` picks the top-similarity band by construction. Kept for
    comparison against real data; use `quantile_thresholds` for synthetic batches.
    """
    y = scorer.fit_labels
    s = scorer.fit_scores
    best_f1, center, _ = _best_f1_threshold(y, s)
    std = float(np.std(s))
    half = margin * std
    return Thresholds(
        center=float(center),
        lower=float(center - half),
        upper=float(center + half),
        margin=margin,
        fit_std=std,
        fit_best_f1=float(best_f1),
        mode="absolute",
    )


def quantile_thresholds(
    scores: np.ndarray,
    *,
    pos_frac: float = 0.3,
    neg_frac: float = 0.3,
) -> Thresholds:
    """Split a batch by its OWN score distribution: top `pos_frac`, bottom `neg_frac`.

    This is the labeling that actually applies to a synthetic batch. It changes
    what a label means, and the change is worth being explicit about: not "good
    by the standard real pairs set" (that threshold doesn't transfer — see
    `compute_thresholds`) but "among the better/worse matches offered to this
    seeker in this batch". Since every candidate was already drawn from the
    top-similarity band, the resulting negatives are hard by construction.
    """
    s = np.asarray(scores, dtype=np.float64)
    upper = float(np.quantile(s, 1.0 - pos_frac))
    lower = float(np.quantile(s, neg_frac))
    if lower >= upper:  # degenerate (tiny batch, or fracs summing past 1)
        lower = upper = float(np.median(s))
    return Thresholds(
        center=float(np.median(s)),
        lower=lower,
        upper=upper,
        margin=0.0,
        fit_std=float(np.std(s)),
        fit_best_f1=float("nan"),
        mode=f"quantile(pos={pos_frac},neg={neg_frac})",
    )


def label_scores(scores: np.ndarray, thresholds: Thresholds) -> list[str | None]:
    """pos above the band, neg below it, None inside it (excluded)."""
    out: list[str | None] = []
    for value in np.asarray(scores, dtype=np.float64):
        if value >= thresholds.upper:
            out.append("pos")
        elif value <= thresholds.lower:
            out.append("neg")
        else:
            out.append(None)
    return out


def pair_texts(candidates) -> tuple[list[str], list[str]]:
    """Seeker/candidate text for a list of select.Candidate, baseline-identical."""
    seeker_texts = [seeker_to_text(c.seeker.profile, c.query) for c in candidates]
    cand_texts = [candidate_to_text(c.candidate.profile) for c in candidates]
    return seeker_texts, cand_texts
