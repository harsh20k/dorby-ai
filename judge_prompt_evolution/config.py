"""Config for the judge-prompt-evolution experiment. Single source of truth
for every knob — new runs are a new ``RunConfig``, not an edited default."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "judge_prompt_evolution"

DEFAULT_OPTIMIZER_MODEL = "anthropic/claude-sonnet-4.5"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

# Real-data examples are sampled from the *train* split only, never holdout —
# this experiment never checks AUC, but if the resulting prompt is later
# scored (a separate, explicitly-confirmed step), the 69-pair holdout must
# still be unseen by anything upstream of that score, exam-style.
DEFAULT_SPLIT = "train"


@dataclass
class RunConfig:
    run_id: str = field(default_factory=lambda: f"run_{time.strftime('%Y%m%d_%H%M%S')}")
    n_iterations: int = 20
    n_positive_examples: int = 2
    n_hard_negative_examples: int = 1
    n_easy_negative_examples: int = 1
    hardness_split: str = "median"  # jaccard >= median negative overlap -> "hard"

    optimizer_model: str = DEFAULT_OPTIMIZER_MODEL
    optimizer_temperature: float = 0.4
    optimizer_max_tokens: int = 8000
    # "openrouter" (default) or "gemini" (direct Google API via GEMINI_API_KEY,
    # bypassing OpenRouter entirely — added when OpenRouter credits ran out
    # mid-evo_007 and to keep the optimizer model consistent with the
    # gemini-3.1-flash-lite AUC-check reference model).
    optimizer_backend: str = "openrouter"

    # Every N optimize-iterations, insert an extra distillation call rather
    # than relying on the meta-prompt's "revise, don't append" instruction
    # alone. 0 disables this (evo_001/evo_002 behavior).
    summarize_every: int = 0
    # "aggressive" = prompts/summarizer.md (evo_004/evo_005 — explicitly
    # pushes toward shortness, which cost it the JSON contract twice at high
    # compression ratios). "gentle" = prompts/summarizer_gentle.md — same
    # goal (clearer, more general principles) but explicitly told length is
    # not the objective; may end up close to its starting size.
    summarizer_variant: str = "aggressive"

    data_dir: Path = DEFAULT_DATA_DIR
    split: str = DEFAULT_SPLIT
    seed: int = 42

    # "naive" = judge_prompt_evolution.seed_prompt.SEED_JUDGE_PROMPT (the
    # 0.6177-AUC prompt this experiment set out to beat). "structured_cot" =
    # baselines.llm_judge.prompt.SYSTEM_PROMPTS["structured_cot"] (imported
    # read-only, not duplicated — we're not forking that prompt's own
    # definition, just using its current text as a different starting point).
    seed_source: str = "naive"

    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR

    push_to_hub: bool = True
    hub_repo: str = "judge-prompt-evolution"
    hub_owner: str | None = None  # falls back to LANGSMITH_PROMPT_OWNER env

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.artifacts_dir = Path(self.artifacts_dir)
        if self.hub_owner is None:
            self.hub_owner = (os.getenv("LANGSMITH_PROMPT_OWNER") or "").strip() or None

    @property
    def run_dir(self) -> Path:
        return self.artifacts_dir / self.run_id

    @property
    def examples_per_iteration(self) -> int:
        return (
            self.n_positive_examples
            + self.n_hard_negative_examples
            + self.n_easy_negative_examples
        )
