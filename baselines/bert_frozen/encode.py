"""Frozen BERT encoder with mean pooling, L2 normalize, and disk cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def l2_normalize(vectors: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return vectors / vectors.norm(p=2, dim=-1, keepdim=True).clamp(min=eps)


def _cache_key(texts: Sequence[str], model_name: str, max_length: int) -> str:
    h = hashlib.sha256()
    h.update(model_name.encode())
    h.update(b"\0")
    h.update(str(max_length).encode())
    h.update(b"\0")
    for text in texts:
        h.update(text.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:32]


class FrozenBertEncoder:
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        device: torch.device | None = None,
        max_length: int = 512,
        cache_dir: Path | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or pick_device()
        self.max_length = max_length
        self.cache_dir = Path(cache_dir) if cache_dir else Path("artifacts/bert_frozen")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.inference_mode()
    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 16,
        show_progress: bool = True,
        cache_name: str | None = None,
    ) -> np.ndarray:
        texts_list = list(texts)
        if not texts_list:
            return np.zeros((0, self.model.config.hidden_size), dtype=np.float32)

        key = cache_name or _cache_key(texts_list, self.model_name, self.max_length)
        cache_path = self.cache_dir / f"emb_{key}.npy"
        meta_path = self.cache_dir / f"emb_{key}.json"

        if cache_path.exists():
            return np.load(cache_path)

        vectors: list[np.ndarray] = []
        indices = range(0, len(texts_list), batch_size)
        if show_progress:
            indices = tqdm(indices, desc=f"encode[{self.device}]", unit="batch")

        for start in indices:
            batch = texts_list[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            outputs = self.model(**encoded)
            pooled = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            pooled = l2_normalize(pooled)
            vectors.append(pooled.detach().cpu().numpy().astype(np.float32))

        matrix = np.vstack(vectors)
        np.save(cache_path, matrix)
        meta_path.write_text(
            json.dumps(
                {
                    "model_name": self.model_name,
                    "max_length": self.max_length,
                    "num_texts": len(texts_list),
                    "dim": int(matrix.shape[1]),
                    "device": str(self.device),
                },
                indent=2,
            )
            + "\n"
        )
        return matrix


def cosine_scores(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine for L2-normalized vectors (elementwise product sum)."""
    if a.ndim == 1:
        a = a[None, :]
    if b.ndim == 1:
        b = b[None, :]
    return np.sum(a * b, axis=-1)
