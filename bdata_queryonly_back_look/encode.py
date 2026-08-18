"""Voyage-4-nano encoder with optional LoRA adapter.

Copied from bdata_voyage_nano_posbg/encode.py. Do not edit that file or
baselines/voyage_nano/. Cosine drift pinned by
tests/test_bdata_queryonly_back_look.py.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def pick_device() -> str:
    """Prefer Apple MPS, then CUDA, else CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def l2_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.clip(norms, eps, None)


def _cache_key(
    texts: Sequence[str],
    model_name: str,
    max_length: int,
    truncate_dim: int | None,
    role: str,
    adapter_dir: str = "",
) -> str:
    h = hashlib.sha256()
    h.update(model_name.encode())
    h.update(b"\0")
    h.update(str(max_length).encode())
    h.update(b"\0")
    h.update(str(truncate_dim).encode())
    h.update(b"\0")
    h.update(role.encode())
    h.update(b"\0")
    h.update(adapter_dir.encode())
    h.update(b"\0")
    for text in texts:
        h.update(text.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:32]


class VoyageNanoEncoder:
    """Frozen local voyage-4-nano with asymmetric query/document encoding + disk cache.

    max_length defaults to 8192 (of 32k supported) to keep MPS memory manageable
    for long Boardy-style dossiers; override with --max-length if needed.
    """

    def __init__(
        self,
        model_name: str = "voyageai/voyage-4-nano",
        device: str | None = None,
        max_length: int = 4096,
        truncate_dim: int | None = 1024,
        cache_dir: Path | None = None,
        adapter_dir: Path | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or pick_device()
        self.max_length = max_length
        self.truncate_dim = truncate_dim
        self.adapter_dir = Path(adapter_dir) if adapter_dir else None
        self.cache_dir = Path(cache_dir) if cache_dir else Path("artifacts/bdata_queryonly_back_look")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"loading {model_name} on {self.device} "
            f"(max_length={max_length}, truncate_dim={truncate_dim}, "
            f"adapter={self.adapter_dir})"
        )
        self.model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            truncate_dim=truncate_dim,
            device=self.device,
        )
        self.model.max_seq_length = max_length
        if self.adapter_dir is not None:
            if not (
                (self.adapter_dir / "adapter_config.json").exists()
                or (self.adapter_dir / "adapter_model.safetensors").exists()
            ):
                raise FileNotFoundError(f"No LoRA adapter at {self.adapter_dir}")
            self.model.load_adapter(str(self.adapter_dir))
            print(f"loaded adapter {self.adapter_dir}")

    def encode(
        self,
        texts: Sequence[str],
        *,
        role: Literal["query", "document"],
        batch_size: int = 4,
        show_progress: bool = True,
        cache_name: str | None = None,
    ) -> np.ndarray:
        texts_list = list(texts)
        if not texts_list:
            if self.truncate_dim is not None:
                dim = self.truncate_dim
            else:
                dim_fn = getattr(self.model, "get_embedding_dimension", None) or getattr(
                    self.model, "get_sentence_embedding_dimension", None
                )
                dim = int(dim_fn() if dim_fn else 1024)
            return np.zeros((0, dim), dtype=np.float32)

        adapter_tag = str(self.adapter_dir) if self.adapter_dir else ""
        key = cache_name or _cache_key(
            texts_list,
            self.model_name,
            self.max_length,
            self.truncate_dim,
            role,
            adapter_tag,
        )
        cache_path = self.cache_dir / f"emb_{key}.npy"
        meta_path = self.cache_dir / f"emb_{key}.json"

        if cache_path.exists():
            return np.load(cache_path)

        encode_fn = self.model.encode_query if role == "query" else self.model.encode_document
        raw = encode_fn(
            texts_list,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        matrix = np.asarray(raw, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        # Belt-and-suspenders: ensure unit L2 even if helper skipped normalize.
        matrix = l2_normalize(matrix).astype(np.float32)

        np.save(cache_path, matrix)
        meta_path.write_text(
            json.dumps(
                {
                    "model_name": self.model_name,
                    "max_length": self.max_length,
                    "truncate_dim": self.truncate_dim,
                    "role": role,
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
    """Row-wise cosine for L2-normalized vectors."""
    if a.ndim == 1:
        a = a[None, :]
    if b.ndim == 1:
        b = b[None, :]
    return np.sum(a * b, axis=-1)
