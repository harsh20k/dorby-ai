"""Build the two independent LoRA towers and a differentiable encode helper.

Not using `SentenceTransformerTrainer` here — it assumes one model being
optimized against one loss over that model's own outputs. This experiment
needs two independently-weighted models optimized jointly against a loss that
mixes both of their outputs (S = s_fwd + lambda*s_rev), which doesn't fit that
shape. `train.py` hand-rolls the training loop instead; this module supplies
the pieces a hand-rolled loop needs (model construction, a differentiable
forward pass with manual prompt prepending, since the Trainer's automatic
per-column prompt handling isn't available outside it).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from twotower.config import TrainConfig
from twotower.train import add_lora_adapter, build_model

Role = Literal["query", "document"]


def build_two_towers(cfg: TrainConfig, device: str) -> tuple[SentenceTransformer, SentenceTransformer, dict[str, Any]]:
    """Ask tower (query-prompt role) and Offer tower (document-prompt role),
    same base model, independent LoRA weights. Matches
    baselines/reciprocal_static's convention: look-text (Ask) always uses the
    query prompt, bg-text (Offer) always uses the document prompt, regardless
    of whether the text belongs to the seeker or candidate role in a given
    call — see docs/reciprocal-static-experiment.md.
    """
    ask_model = build_model(cfg, device)
    ask_info = add_lora_adapter(ask_model, cfg)

    offer_model = build_model(cfg, device)
    offer_info = add_lora_adapter(offer_model, cfg)

    return ask_model, offer_model, {"ask": ask_info, "offer": offer_info}


def _prompt_for(cfg: TrainConfig, role: Role) -> str:
    return cfg.query_prompt if role == "query" else cfg.document_prompt


def encode_texts(
    model: SentenceTransformer,
    texts: list[str],
    *,
    role: Role,
    cfg: TrainConfig,
    device: torch.device,
    normalize: bool = True,
) -> torch.Tensor:
    """Differentiable forward pass: tokenize -> model(features) ->
    sentence_embedding. Prompt is prepended to the raw text manually (the
    Trainer's `prompts=` mechanism only applies inside
    SentenceTransformerTrainer, which this package doesn't use)."""
    prompt = _prompt_for(cfg, role)
    prefixed = [prompt + t for t in texts]
    preprocess = getattr(model, "preprocess", None) or model.tokenize
    features = preprocess(prefixed)
    features = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in features.items()}
    emb = model(features)["sentence_embedding"]
    if normalize:
        emb = F.normalize(emb, p=2, dim=-1)
    return emb


@torch.no_grad()
def encode_batched(
    model: SentenceTransformer,
    texts: list[str],
    *,
    role: Role,
    cfg: TrainConfig,
    device: torch.device,
    batch_size: int = 8,
) -> np.ndarray:
    """Inference-only batched encode (no grad) — used by eval_dev.py and
    eval.py, where the model is fixed and only the output array is needed."""
    was_training = model.training
    model.eval()
    chunks = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        emb = encode_texts(model, chunk, role=role, cfg=cfg, device=device)
        chunks.append(emb.cpu().numpy())
    if was_training:
        model.train()
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 0), dtype=np.float32)


def trainable_parameters(*models: SentenceTransformer) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    for m in models:
        params.extend(p for p in m.parameters() if p.requires_grad)
    return params
