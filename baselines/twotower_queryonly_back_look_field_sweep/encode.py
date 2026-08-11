"""Disk-cached encoder wrapper around the fine-tuned ``queryonly_back_look_001`` checkpoint.

``queryonly_back_look_001`` (``twotower_queryonly_back_look``) is, as of this
experiment, the new best two-tower fine-tune in the project on every tracked
metric (all-200: pair AUC 0.5983, hard-neg AUC 0.6564, MRR 0.4791, R@1 0.30 —
beating ``top1_ctrl``'s 0.5683 / 0.5484 / 0.3550 / 0.19 on every metric). Same
LoRA shape as every other checkpoint in this project (rank 8, alpha 16,
dropout 0.05, q/k/v/o_proj on frozen Voyage-4-nano), but trained on a
different text representation: seeker = search-query-only (no profile
fields), candidate = background+lookingFor — the field/query sweep's own
recall@1-best combo, found frozen against ``top1_ctrl`` first and only then
actually trained on (`docs/twotower-queryonly-back-look-experiment.md`).

Reuses ``twotower/eval.py::load_model_for_eval`` (adapter loading via
SentenceTransformer's native ``.load_adapter()``) and ``encode_role``
(``encode_query``/``encode_document`` dispatch) unmodified and read-only —
those are generic model-loading infra, not part of what this experiment
varies. The only thing added here is the disk-cache-by-``cache_name``
wrapper needed for the same encoding-dedup trick
``twotower_top1_ctrl_field_sweep/encode.py`` uses (22 unique encode groups
instead of 105 full passes).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

from baselines.voyage_nano.encode import l2_normalize
from twotower.eval import encode_role, load_model_for_eval

QUERYONLY_BACK_LOOK_ADAPTER_DIR = Path(
    "artifacts/twotower_queryonly_back_look/queryonly_back_look_001/adapter"
)
QUERYONLY_BACK_LOOK_BASE_MODEL = "voyageai/voyage-4-nano"
QUERYONLY_BACK_LOOK_MAX_SEQ_LENGTH = 4096  # matches twotower/config.py's TrainConfig default
QUERYONLY_BACK_LOOK_TRUNCATE_DIM = 1024


class QueryonlyBackLookEncoder:
    """Loads voyage-4-nano + the queryonly_back_look LoRA adapter once; caches encodes to disk."""

    def __init__(
        self,
        *,
        adapter_dir: Path = QUERYONLY_BACK_LOOK_ADAPTER_DIR,
        model_name: str = QUERYONLY_BACK_LOOK_BASE_MODEL,
        device: str,
        max_seq_length: int = QUERYONLY_BACK_LOOK_MAX_SEQ_LENGTH,
        truncate_dim: int = QUERYONLY_BACK_LOOK_TRUNCATE_DIM,
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
            dim = self.model.get_sentence_embedding_dimension() or QUERYONLY_BACK_LOOK_TRUNCATE_DIM
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
