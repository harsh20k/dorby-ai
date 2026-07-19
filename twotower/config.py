"""Hyperparameters and paths for the two-tower LoRA fine-tune."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_QUERY_PROMPT = "Represent the query for retrieving supporting documents: "
DEFAULT_DOCUMENT_PROMPT = "Represent the document for retrieval: "


@dataclass(frozen=True)
class TrainConfig:
    """Single source of truth for training + Modal runs."""

    # Model
    model_name: str = "voyageai/voyage-4-nano"
    model_revision: str | None = None  # pin on first successful download
    trust_remote_code: bool = True
    truncate_dim: int = 1024
    max_seq_length: int = 4096

    # LoRA
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    expected_layers_per_target: int = 12

    # Optimization
    epochs: int = 5
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    train_batch_size: int = 2
    eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    seed: int = 42

    # Data
    data_dir: Path = Path("data")
    split_path: Path = Path("data/synthetic/seed_split.json")
    train_dev_user_fraction: float = 0.1
    train_dev_min_pairs: int = 20
    contrastive_margin: float = 0.5

    # Prompts (Voyage asymmetric convention)
    query_prompt: str = DEFAULT_QUERY_PROMPT
    document_prompt: str = DEFAULT_DOCUMENT_PROMPT

    # Artifacts
    output_dir: Path = Path("artifacts/twotower/run_001")
    save_total_limit: int = 3
    logging_steps: int = 10

    # Eval
    primary_metric: str = "pair_auc"
    run_holdout: bool = False

    # Misc
    # voyage-4-nano remote code currently rejects gradient checkpointing.
    gradient_checkpointing: bool = False
    bf16: bool | None = None  # None = auto from CUDA capability
    fp16: bool | None = None

    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["data_dir"] = str(self.data_dir)
        payload["split_path"] = str(self.split_path)
        payload["output_dir"] = str(self.output_dir)
        payload["lora_target_modules"] = list(self.lora_target_modules)
        return payload


def resolve_mixed_precision(device: str, cfg: TrainConfig) -> tuple[bool, bool]:
    """Return (bf16, fp16) based on config overrides and GPU capability."""
    if cfg.bf16 is not None or cfg.fp16 is not None:
        bf16 = bool(cfg.bf16) if cfg.bf16 is not None else False
        fp16 = bool(cfg.fp16) if cfg.fp16 is not None else False
        if bf16 and fp16:
            raise ValueError("bf16 and fp16 cannot both be True")
        return bf16, fp16

    if device != "cuda":
        return False, False

    import torch

    major, _minor = torch.cuda.get_device_capability()
    # Ampere+ (sm80) has native BF16; older GPUs (T4=sm75) use FP16.
    if major >= 8:
        return True, False
    return False, True
