"""Runtime config for the synthetic pair pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_SPLIT_PATH = DEFAULT_DATA_DIR / "synthetic" / "seed_split.json"
DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "synth"

PROFILE_KEYS = (
    "positioning",
    "lookingFor",
    "introPreferences",
    "personalPreferences",
    "meetingAndSchedulingPreferences",
    "background",
    "locationAvailability",
    "notes",
)

PAIR_KEYS = (
    "userContactId",
    "matchContactId",
    "userContactFileVersion",
    "matchContactFileVersion",
    "searchQuery",
    "userContactFile",
    "matchContactFile",
)

FAILURE_MODES = (
    "wrong_side",
    "wrong_stage",
    "wrong_role",
    "geo_mismatch",
    "prefs_conflict",
)

MAX_RETRIES = 2
HOLDOUT_FRACTION = 0.2
FEW_SHOT_K = 2
ID_LENGTH = 25
ID_PREFIX = "cm"


@dataclass
class PipelineConfig:
    data_dir: Path = DEFAULT_DATA_DIR
    split_path: Path = DEFAULT_SPLIT_PATH
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR
    generate_model: str = field(
        default_factory=lambda: os.getenv("SYNTH_GENERATE_MODEL", "gpt-4.1-mini")
    )
    judge_model: str = field(
        default_factory=lambda: os.getenv("SYNTH_JUDGE_MODEL", "gpt-4.1")
    )
    temperature: float = 0.7
    judge_temperature: float = 0.0
    few_shot_k: int = FEW_SHOT_K
    max_retries: int = MAX_RETRIES
    holdout_fraction: float = HOLDOUT_FRACTION
    dry_run: bool = False
    prompt_version: str = "v1"

    @property
    def positive_path(self) -> Path:
        return self.data_dir / "dataset_positive.json"

    @property
    def negative_path(self) -> Path:
        return self.data_dir / "dataset_negative.json"
