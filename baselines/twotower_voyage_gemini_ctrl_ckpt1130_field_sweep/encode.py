"""Disk-cached encoder wrapper around the raw ``checkpoint-1130`` (final epoch).

Same training run as ``twotower_voyage_gemini_ctrl_field_sweep``
(`top1_ctrl`'s exact LoRA recipe, retrained on ``pairing_voyage_gemini/
smoke_test_002``), but a different point in it: the final-epoch checkpoint
(step 1130/1130) instead of the recall@1-best checkpoint
(step 452/1130) the training pipeline actually selected as ``adapter``.
See this package's ``__init__.py`` for why this comparison matters.

Reuses ``twotower/eval.py::load_model_for_eval`` (adapter loading via
SentenceTransformer's native ``.load_adapter()``) and ``encode_role``
(``encode_query``/``encode_document`` dispatch) unmodified and read-only —
those are generic model-loading infra, not part of what this experiment
varies. The only thing added here is the disk-cache-by-``cache_name``
wrapper needed for the same encoding-dedup trick every prior field/query
sweep in this project uses (22 unique encode groups instead of 105 full
passes).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

from baselines.voyage_nano.encode import l2_normalize
from twotower.eval import encode_role, load_model_for_eval

VOYAGE_GEMINI_CTRL_CKPT1130_ADAPTER_DIR = Path(
    "artifacts/twotower_voyage_gemini_ctrl/voyage_gemini_ctrl_001_checkpoint1130/adapter"
)
VOYAGE_GEMINI_CTRL_CKPT1130_BASE_MODEL = "voyageai/voyage-4-nano"
VOYAGE_GEMINI_CTRL_CKPT1130_MAX_SEQ_LENGTH = 4096  # matches twotower/config.py's TrainConfig default
VOYAGE_GEMINI_CTRL_CKPT1130_TRUNCATE_DIM = 1024


class VoyageGeminiCtrlCkpt1130Encoder:
    """Loads voyage-4-nano + the voyage_gemini_ctrl LoRA adapter once; caches encodes to disk."""

    def __init__(
        self,
        *,
        adapter_dir: Path = VOYAGE_GEMINI_CTRL_CKPT1130_ADAPTER_DIR,
        model_name: str = VOYAGE_GEMINI_CTRL_CKPT1130_BASE_MODEL,
        device: str,
        max_seq_length: int = VOYAGE_GEMINI_CTRL_CKPT1130_MAX_SEQ_LENGTH,
        truncate_dim: int = VOYAGE_GEMINI_CTRL_CKPT1130_TRUNCATE_DIM,
        cache_dir: Path,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"loading {model_name} + adapter {adapter_dir} on {device} "
            f"(max_seq_length={max_seq_length}, truncate_dim={truncate_dim})"
        )
        self.model = load_model_for_eval(
            model_name=model_name,
            adapter_dir=Path(adapter_dir),
            device=device,
            max_seq_length=max_seq_length,
            truncate_dim=truncate_dim,
        )

    def encode(
        self,
        texts: Sequence[str],
        *,
        role: Literal["query", "document"],
        batch_size: int = 4,
        cache_name: str,
    ) -> np.ndarray:
        texts_list = list(texts)
        cache_path = self.cache_dir / f"emb_{cache_name}.npy"
        meta_path = self.cache_dir / f"emb_{cache_name}.json"
        if not texts_list:
            dim = self.model.get_sentence_embedding_dimension() or VOYAGE_GEMINI_CTRL_CKPT1130_TRUNCATE_DIM
            return np.zeros((0, dim), dtype=np.float32)
        if cache_path.exists():
            return np.load(cache_path)

        matrix = encode_role(self.model, texts_list, role=role, batch_size=batch_size)
        matrix = l2_normalize(np.asarray(matrix, dtype=np.float32)).astype(np.float32)

        np.save(cache_path, matrix)
        meta_path.write_text(
            json.dumps(
                {"role": role, "num_texts": len(texts_list), "dim": int(matrix.shape[1])},
                indent=2,
            )
            + "\n"
        )
        return matrix


def cosine_scores(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.ndim == 1:
        a = a[None, :]
    if b.ndim == 1:
        b = b[None, :]
    return np.sum(a * b, axis=-1)
