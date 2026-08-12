"""Config for the isolated B-data Voyage-4-nano Accept/Reject experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ExperimentConfig:
    source_path: Path = ROOT / "data" / "B-data.json"
    # Matched copy of bdata_tfidf/split.json (same seeker-disjoint freeze).
    split_path: Path = ROOT / "bdata_voyage_nano" / "split.json"
    artifacts_dir: Path = ROOT / "artifacts" / "bdata_voyage_nano"

    # Match baselines/voyage_nano documented defaults.
    model_name: str = "voyageai/voyage-4-nano"
    batch_size: int = 4
    max_length: int = 8192
    truncate_dim: int | None = 1024

    # Within-seeker ranking: only headline if enough dual-label seekers
    min_within_seeker_n: int = 30


DEFAULT_CONFIG = ExperimentConfig()
