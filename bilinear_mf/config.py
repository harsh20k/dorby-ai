"""Single source of truth for this experiment's knobs and paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parent.parent

Backbone = Literal["tfidf", "voyage_large"]

ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "bilinear_mf"


@dataclass(frozen=True)
class BilinearConfig:
    """Everything a run needs. Serialized into each run's ``run_meta.json``."""

    run_id: str = "mf_001"

    # --- population -------------------------------------------------------
    data_dir: Path = REPO_ROOT / "data"
    split_path: Path = REPO_ROOT / "data" / "synthetic" / "seed_split.json"

    # --- frozen backbone the bilinear arm sits on top of ------------------
    backbone: Backbone = "voyage_large"
    # TF-IDF vectorizer settings, matched to baselines/tfidf/eval.py defaults so
    # the `lsa` arm's rank-full limit reproduces the published TF-IDF row.
    max_features: int = 20000
    ngram_range: tuple[int, int] = (1, 2)
    voyage_model: str = "voyage-4-large"
    voyage_output_dimension: int = 1024

    # --- dimensionality reduction ----------------------------------------
    # Label-free truncated SVD applied to the backbone matrix before the
    # bilinear head. On TF-IDF this *is* LSA; on Voyage it is plain PCA-style
    # compression, used only to keep the head's parameter count sane.
    reduce_dim: int = 128

    # --- bilinear head ----------------------------------------------------
    rank: int = 16
    init_scale: float = 1e-3
    weight_decay: float = 1e-2
    lr: float = 5e-2
    steps: int = 400
    seed: int = 17

    # --- protocol ---------------------------------------------------------
    # Ranks swept by the `lsa` arm.
    lsa_ranks: tuple[int, ...] = (16, 32, 64, 128, 256, 512)
    # Hyperparameter grid searched by inner CV on the *train split only*.
    grid_ranks: tuple[int, ...] = (4, 8, 16, 32)
    grid_weight_decay: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0)
    # Swept, not fixed: the SVD width changes whether the head survives
    # regularization at all (at d=64 on TF-IDF it does, at d=128 it collapses to
    # zero), so leaving it outside the grid would make the reported number a
    # choice made by looking at the eval set.
    grid_reduce_dims: tuple[int, ...] = (32, 64, 128, 256)
    # Repeats of the label-permutation null used as this experiment's noise floor.
    n_permutations: int = 50

    extra: dict = field(default_factory=dict)

    @property
    def run_dir(self) -> Path:
        return ARTIFACTS_DIR / self.run_id

    def to_json(self) -> dict:
        return {
            "run_id": self.run_id,
            "backbone": self.backbone,
            "max_features": self.max_features,
            "ngram_range": list(self.ngram_range),
            "voyage_model": self.voyage_model,
            "voyage_output_dimension": self.voyage_output_dimension,
            "reduce_dim": self.reduce_dim,
            "rank": self.rank,
            "init_scale": self.init_scale,
            "weight_decay": self.weight_decay,
            "lr": self.lr,
            "steps": self.steps,
            "seed": self.seed,
            "lsa_ranks": list(self.lsa_ranks),
            "grid_ranks": list(self.grid_ranks),
            "grid_weight_decay": list(self.grid_weight_decay),
            "grid_reduce_dims": list(self.grid_reduce_dims),
            "n_permutations": self.n_permutations,
        }
