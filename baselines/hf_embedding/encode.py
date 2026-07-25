"""Generic open-weight HF embedding encoder via sentence-transformers.

Generalizes baselines/voyage_nano/encode.py to arbitrary HF model ids: instead
of Voyage's bespoke encode_query/encode_document methods, uses the standard
sentence-transformers `model.encode(texts, prompt_name=...)` API that most
modern open-weight embedders (Qwen3-Embedding, GTE, E5, BGE, Arctic Embed,
mxbai, ...) support natively. Per-model quirks (prompt name, trust_remote_code,
Matryoshka truncation) come from models.py's MODEL_REGISTRY.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from baselines.hf_embedding.models import get_model_spec, slugify
from baselines.voyage_nano.encode import cosine_scores, l2_normalize, pick_device

__all__ = [
    "HFEmbeddingEncoder",
    "FlagICLEncoder",
    "get_encoder_class",
    "cosine_scores",
    "l2_normalize",
    "pick_device",
]

QUERY_INSTRUCTION = (
    "Given a search query and a networking user's profile, retrieve candidate "
    "contact profiles that would make a good introduction match."
)


def _resolve_dtype(dtype: str, device: str) -> torch.dtype | None:
    if dtype == "auto":
        return torch.bfloat16 if device == "cuda" else torch.float32
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]


def _cache_key(
    texts: Sequence[str],
    model_name: str,
    max_length: int,
    truncate_dim: int | None,
    role: str,
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
    for text in texts:
        h.update(text.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:32]


class HFEmbeddingEncoder:
    """Frozen open-weight HF embedding model with asymmetric query/document
    encoding (where the model's registry entry declares a query prompt) +
    disk cache, following the same cache-file layout as VoyageNanoEncoder."""

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        max_length: int = 8192,
        truncate_dim: int | None = None,
        dtype: str = "auto",
        cache_dir: Path | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or pick_device()
        self.max_length = max_length
        self.spec = get_model_spec(model_name)
        self.truncate_dim = truncate_dim if self.spec.supports_truncate_dim else None
        self.cache_dir = (
            Path(cache_dir) if cache_dir else Path("artifacts/hf_embedding") / slugify(model_name)
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        resolved_dtype = _resolve_dtype(dtype, self.device)
        print(
            f"loading {model_name} on {self.device} (dtype={resolved_dtype}, "
            f"max_length={max_length}, truncate_dim={self.truncate_dim}, "
            f"trust_remote_code={self.spec.trust_remote_code})"
        )
        self.model = SentenceTransformer(
            model_name,
            trust_remote_code=self.spec.trust_remote_code,
            truncate_dim=self.truncate_dim,
            device=self.device,
            model_kwargs={"torch_dtype": resolved_dtype},
        )
        self.model.max_seq_length = max_length

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
            dim = self.truncate_dim or self.model.get_sentence_embedding_dimension()
            return np.zeros((0, dim), dtype=np.float32)

        key = cache_name or _cache_key(
            texts_list, self.model_name, self.max_length, self.truncate_dim, role
        )
        cache_path = self.cache_dir / f"emb_{key}.npy"
        meta_path = self.cache_dir / f"emb_{key}.json"

        if cache_path.exists():
            return np.load(cache_path)

        prompt_name = self.spec.query_prompt_name if role == "query" else None
        raw = self.model.encode(
            texts_list,
            prompt_name=prompt_name,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        matrix = np.asarray(raw, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        matrix = l2_normalize(matrix).astype(np.float32)

        np.save(cache_path, matrix)
        meta_path.write_text(
            json.dumps(
                {
                    "model_name": self.model_name,
                    "max_length": self.max_length,
                    "truncate_dim": self.truncate_dim,
                    "role": role,
                    "prompt_name": prompt_name,
                    "num_texts": len(texts_list),
                    "dim": int(matrix.shape[1]),
                    "device": str(self.device),
                },
                indent=2,
            )
            + "\n"
        )
        return matrix


class FlagICLEncoder:
    """BAAI/bge-en-icl via BAAI's own FlagEmbedding library (FlagICLModel) —
    not sentence-transformers-loadable. Same encode() interface and cache-file
    layout as HFEmbeddingEncoder so eval.py doesn't need to branch on it.
    Runs zero-shot (no in-context examples), matching how every other baseline
    here is evaluated."""

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        max_length: int = 8192,
        truncate_dim: int | None = None,
        dtype: str = "auto",
        cache_dir: Path | None = None,
    ) -> None:
        from FlagEmbedding import FlagICLModel

        self.model_name = model_name
        self.device = device or pick_device()
        self.max_length = max_length
        self.truncate_dim = None  # no Matryoshka support
        self.cache_dir = (
            Path(cache_dir) if cache_dir else Path("artifacts/hf_embedding") / slugify(model_name)
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"loading {model_name} via FlagICLModel on {self.device} (max_length={max_length})")
        self.model = FlagICLModel(
            model_name,
            query_instruction_for_retrieval=QUERY_INSTRUCTION,
            examples_for_task=None,  # zero-shot, no few-shot examples
            devices=[self.device],
            use_fp16=(self.device == "cuda"),
        )

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
            return np.zeros((0, 4096), dtype=np.float32)

        key = cache_name or _cache_key(
            texts_list, self.model_name, self.max_length, self.truncate_dim, role
        )
        cache_path = self.cache_dir / f"emb_{key}.npy"
        meta_path = self.cache_dir / f"emb_{key}.json"

        if cache_path.exists():
            return np.load(cache_path)

        encode_fn = self.model.encode_queries if role == "query" else self.model.encode_corpus
        raw = encode_fn(texts_list, batch_size=batch_size, max_length=self.max_length)
        matrix = l2_normalize(np.asarray(raw, dtype=np.float32))

        np.save(cache_path, matrix)
        meta_path.write_text(
            json.dumps(
                {
                    "model_name": self.model_name,
                    "max_length": self.max_length,
                    "truncate_dim": None,
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


def get_encoder_class(model_name: str):
    """Pick HFEmbeddingEncoder or FlagICLEncoder based on the model's registry loader."""
    spec = get_model_spec(model_name)
    if spec.loader == "flagembedding_icl":
        return FlagICLEncoder
    return HFEmbeddingEncoder
