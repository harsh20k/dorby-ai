"""Model loading with an explicit weight dtype, for large backbones.

twotower.train.build_model() (reused everywhere else in this package) never
sets torch_dtype, so it loads in fp32 — fine for voyage-4-nano (347M params,
~1.4GB fp32) but OOMs an 8B model (32GB fp32 vs. 16GB bf16) even at batch
size 1 on a 40GB A100. This mirrors baselines/hf_embedding/encode.py's
existing, already-proven bf16 loading path for the same model family,
without modifying twotower/train.py.
"""

from __future__ import annotations

import torch
from sentence_transformers import SentenceTransformer

from twotower.config import TrainConfig


def build_model_with_dtype(
    cfg: TrainConfig,
    device: str,
    *,
    torch_dtype: torch.dtype | None = None,
    gradient_checkpointing: bool = False,
) -> SentenceTransformer:
    model_kwargs = {"torch_dtype": torch_dtype} if torch_dtype is not None else {}
    model = SentenceTransformer(
        cfg.model_name,
        trust_remote_code=cfg.trust_remote_code,
        truncate_dim=cfg.truncate_dim,
        device=device,
        revision=cfg.model_revision,
        prompts={"query": cfg.query_prompt, "document": cfg.document_prompt},
        model_kwargs=model_kwargs,
    )
    model.max_seq_length = cfg.max_seq_length
    if gradient_checkpointing:
        try:
            auto = model[0].auto_model
            if hasattr(auto, "gradient_checkpointing_enable"):
                auto.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            if hasattr(auto, "config"):
                auto.config.use_cache = False
        except Exception as exc:  # noqa: BLE001 — best-effort
            print(f"warning: gradient checkpointing not enabled: {exc}")
    return model
