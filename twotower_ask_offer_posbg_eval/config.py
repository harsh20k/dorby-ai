"""Config for the ask_offer_001 pos+bg eval-time field swap."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ExperimentConfig:
    data_dir: Path = ROOT / "data"
    split_path: Path = ROOT / "data" / "synthetic" / "seed_split.json"
    artifacts_dir: Path = ROOT / "artifacts" / "twotower_ask_offer_posbg_eval"
    # Adapters live on Modal volume; local path is only used if already pulled.
    adapter_dir: Path = ROOT / "artifacts" / "twotower_ask_offer" / "ask_offer_001" / "adapter"
    lam: float = 1.75  # same fixed lambda the towers were trained with
    batch_size: int = 8
    source_run_id: str = "ask_offer_001"


DEFAULT_CONFIG = ExperimentConfig()
