"""Runtime config for the synthetic pair pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_SPLIT_PATH = DEFAULT_DATA_DIR / "synthetic" / "seed_split.json"
DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "synth"

# OpenRouter (OpenAI-compatible). DeepSeek is cheap and good enough for synth.
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_GENERATE_MODEL = "deepseek/deepseek-v4-pro"
DEFAULT_JUDGE_MODEL = "google/gemini-3.1-flash-lite"


def load_dotenv() -> None:
    """Load repo-root `.env` if python-dotenv is installed (no-op otherwise)."""
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv as _load

        _load(env_path, override=False)
    except ImportError:
        # Minimal fallback: KEY=VALUE lines only
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = value


load_dotenv()

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
    # OpenRouter API key preferred; falls back to OPENAI_API_KEY for local proxies.
    api_key: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or ""
    )
    base_url: str = field(
        default_factory=lambda: os.getenv(
            "OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL
        )
    )
    generate_model: str = field(
        default_factory=lambda: os.getenv(
            "SYNTH_GENERATE_MODEL", DEFAULT_GENERATE_MODEL
        )
    )
    judge_model: str = field(
        default_factory=lambda: os.getenv("SYNTH_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
    )
    temperature: float = 0.7
    judge_temperature: float = 0.0
    few_shot_k: int = FEW_SHOT_K
    max_retries: int = MAX_RETRIES
    holdout_fraction: float = HOLDOUT_FRACTION
    dry_run: bool = False
    # Fallback label when hub refs are unavailable; overridden at runtime by
    # hub commit/tag when prompts are pulled (see synth_pipeline.prompts).
    prompt_version: str = field(
        default_factory=lambda: os.getenv("SYNTH_PROMPT_VERSION", "v1")
    )

    @property
    def positive_path(self) -> Path:
        return self.data_dir / "dataset_positive.json"

    @property
    def negative_path(self) -> Path:
        return self.data_dir / "dataset_negative.json"
