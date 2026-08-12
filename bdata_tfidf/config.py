"""Config for the isolated B-data TF-IDF Accept/Reject experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ExperimentConfig:
    source_path: Path = ROOT / "data" / "B-data.json"
    split_path: Path = ROOT / "bdata_tfidf" / "split.json"
    artifacts_dir: Path = ROOT / "artifacts" / "bdata_tfidf"

    # TF-IDF knobs — match baselines/tfidf defaults
    max_features: int = 20000
    ngram_min: int = 1
    ngram_max: int = 2
    min_df: int = 1

    # Seeker-disjoint freeze
    holdout_frac: float = 0.30
    split_seed: int = 42

    # Within-seeker ranking: only report if enough dual-label seekers
    min_within_seeker_n: int = 30


DEFAULT_CONFIG = ExperimentConfig()
