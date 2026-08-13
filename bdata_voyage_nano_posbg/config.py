"""Config for the isolated B-data Voyage-4-nano posbg experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ExperimentConfig:
    source_path: Path = ROOT / "data" / "B-data.json"
    unique_contacts_path: Path = ROOT / "data" / "unique_contacts_B_data.json"
    artifacts_dir: Path = ROOT / "artifacts" / "bdata_voyage_nano_posbg"

    model_name: str = "voyageai/voyage-4-nano"
    batch_size: int = 4
    max_length: int = 8192
    truncate_dim: int | None = 1024

    min_within_seeker_n: int = 30
    retrieval_query_batch_size: int = 256
    retrieval_ks: tuple[int, ...] = (1, 5, 10, 50, 100)


DEFAULT_CONFIG = ExperimentConfig()
