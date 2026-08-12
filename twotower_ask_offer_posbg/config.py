"""Config for the ask/offer two-tower reciprocal fine-tune.

Both towers share the exact same LoRA/optimization hyperparameters as
`voyage_gemini_ctrl_001` (the fine-tune this experiment is compared against
in the training plan's side-by-side table) so the only variables that change
are: two independently-trained towers instead of one, and a loss computed on
the combined reciprocal score S instead of s_fwd alone. One `TrainConfig`
instance is reused to build both towers — `build_model`/`add_lora_adapter`
(twotower.train, imported read-only) don't read `output_dir`, and every other
field (model name, LoRA rank/targets, prompts, max_seq_length) is identical
between the two towers by design, so a single shared config is correct, not
a shortcut.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from twotower.config import TrainConfig

MODEL_NAME = "voyageai/voyage-4-nano"

# Matches twotower_voyage_gemini_ctrl's MODEL_PRESETS["voyage-4-nano"] exactly
# (rank/alpha/dropout/targets/batch/grad-accum/lr) so the two experiments are
# comparable on everything except tower count + loss.
TOWER_CONFIG = TrainConfig(
    model_name=MODEL_NAME,
    trust_remote_code=True,
    truncate_dim=1024,
    max_seq_length=4096,
    lora_rank=8,
    lora_alpha=16,
    lora_dropout=0.05,
    lora_target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
    expected_layers_per_target=12,
    train_batch_size=6,
    eval_batch_size=6,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
)

QUERY_PROMPT = "Represent the query for retrieving supporting documents: "
DOCUMENT_PROMPT = "Represent the document for retrieval: "

EFFECTIVE_BATCH_TARGET = 12


@dataclass(frozen=True)
class AskOfferConfig:
    run_id: str
    rows_path: Path = Path("artifacts/twotower_ask_offer_posbg/ask_offer_rows.json")
    output_dir: Path = Path("artifacts/twotower_ask_offer_posbg")

    # Combined score: S = s_fwd + lambda * s_rev. Fixed hyperparameter, never
    # trained jointly with the towers — see the training plan's "lambda:
    # fixed, not learned" section for why (collapse-to-zero risk, the
    # bilinear_mf precedent). Default is the value already fit on real data
    # with zero training (reciprocal_static, lambda=1.75).
    lam: float = 1.75

    epochs: int = 5
    negatives_per_anchor: int = 1
    seed: int = 42
    loss_scale: float = 20.0  # matches MultipleNegativesRankingLoss's library default

    dev_seeker_fraction: float = 0.1
    dev_min_rows: int = 20

    tower: TrainConfig = field(default_factory=lambda: TOWER_CONFIG)

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.run_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "rows_path": str(self.rows_path),
            "output_dir": str(self.output_dir),
            "lam": self.lam,
            "epochs": self.epochs,
            "negatives_per_anchor": self.negatives_per_anchor,
            "seed": self.seed,
            "loss_scale": self.loss_scale,
            "dev_seeker_fraction": self.dev_seeker_fraction,
            "dev_min_rows": self.dev_min_rows,
            "tower": self.tower.to_dict(),
        }


def build_config(run_id: str, **overrides: Any) -> AskOfferConfig:
    cfg = AskOfferConfig(run_id=run_id, **overrides)
    eff = cfg.tower.train_batch_size * cfg.tower.gradient_accumulation_steps
    if eff != EFFECTIVE_BATCH_TARGET:
        print(
            f"\n{'!' * 78}\nWARNING: effective batch is "
            f"{cfg.tower.train_batch_size}x{cfg.tower.gradient_accumulation_steps}={eff}, "
            f"not {EFFECTIVE_BATCH_TARGET}. Not comparable to voyage_gemini_ctrl_001 "
            f"(different optimizer-step count).\n{'!' * 78}\n"
        )
    return cfg
