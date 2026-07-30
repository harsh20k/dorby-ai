"""Configuration for the multi-gate mixture-of-experts re-ranker.

Single source of truth for every knob, mirroring ``twotower/config.py``'s role.
Defaults here are deliberately small: this model is fit on **111 real training
pairs**, which is the binding constraint on every architectural choice. See
``model.py`` for why the expert/gate sizes are what they are.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MoEConfig:
    # ---- data ----
    data_dir: Path = Path("data")
    split_path: Path = Path("data/synthetic/seed_split.json")
    nano_artifacts: Path = Path("artifacts/voyage_nano")
    judge_verdicts: Path = Path(
        "artifacts/llm_judge/openrouter_google_gemini_3_1_flash_lite_naive/verdicts.json"
    )
    #: Real pairs only. Promoted synthetic pairs are excluded by default because
    #: `docs/possible-bugs.md` #4 showed the old generator's artifacts were
    #: actively harmful, and no fixed-generator batch has been promoted yet.
    include_synth: bool = False

    # ---- features ----
    #: Optional PCA-reduced embedding features on top of the hand-built ones.
    #: Off by default: raw 1024-d embeddings would put thousands of parameters
    #: against 111 examples. Fitted on train only when enabled.
    emb_pca_dims: int = 0

    # ---- architecture ----
    n_experts: int = 3
    expert_hidden: int = 4
    #: Gate temperature (the professor's Idea 1). Lower = more polarized routing.
    #: 0.05 is the value that won the section-aggregation sweep; note that sweep
    #: also found tau -> 0 (hard argmax) was *worse*, so do not assume sharper
    #: is better.
    tau: float = 0.05
    #: Randomly zero this fraction of experts per step. Cheap regularization and
    #: a direct defence against one expert dominating.
    expert_dropout: float = 0.2

    # ---- tasks ----
    #: Task order is fixed: index 0 is the real objective, index 1 the auxiliary.
    task_names: tuple[str, ...] = ("accept", "judge")
    #: Auxiliary task is downweighted: it is a *related* target, not the truth.
    task_weights: tuple[float, ...] = (1.0, 0.3)

    # ---- gate regularization (both terms, they pull against each other) ----
    #: Minimize per-example gate entropy -> each pair commits to an expert.
    sharpen_weight: float = 0.05
    #: Maximize entropy of the batch-average gate -> experts stay balanced.
    #: Without this, `sharpen_weight` alone collapses every example onto one
    #: expert, which is exactly what Diagnostic 1 detects.
    balance_weight: float = 0.10

    # ---- optimization ----
    epochs: int = 200
    lr: float = 3e-3
    weight_decay: float = 1e-2
    batch_size: int = 32
    seed: int = 42
    #: Early-stopping metric, measured on train-dev (never holdout).
    early_stop_patience: int = 40

    # ---- output ----
    run_id: str = "moe_001"
    out_dir: Path = Path("artifacts/moe_reranker")

    def run_dir(self) -> Path:
        return self.out_dir / self.run_id

    def as_json(self) -> dict:
        out = {}
        for k, v in self.__dict__.items():
            out[k] = str(v) if isinstance(v, Path) else v
        return out
