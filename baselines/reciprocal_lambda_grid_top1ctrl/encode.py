"""Disk-cached encoder wrapper around the fine-tuned ``top1_ctrl`` checkpoint.

Deliberate duplicate of a wrapper built on another branch
(``worktree-llm-judge-experiment``, ``baselines/twotower_top1_ctrl_field_sweep/
encode.py``) — copied rather than imported cross-branch so this experiment's
numbers stay reproducible from ``main`` alone (experiment isolation rule).

``top1_ctrl_001`` (``twotower_top1_optimised``) is this project's best
fine-tuned model on the metric/population CLAUDE.md treats as authoritative
— a LoRA adapter (rank 8, q/k/v/o_proj) on top of frozen Voyage-4-nano,
trained with micro-batch 6 and a single negative per anchor (the winning
cell in "Which Lever Actually Worked?", `docs/twotower-rrf-triplet-ablation-experiment.md`).
Holdout: pair AUC 0.5974, hard-neg AUC 0.6121, MRR 0.5436.

Reuses ``twotower/eval.py::load_model_for_eval`` (adapter loading via
SentenceTransformer's native ``.load_adapter()``) and ``encode_role``
(``encode_query``/``encode_document`` dispatch) unmodified and read-only —
those are generic model-loading infra already on ``main``, not part of what
this experiment varies. The only thing added here is a disk-cache-by-
``cache_name`` wrapper, matching the shape ``baselines/voyage_nano/encode.py::
VoyageNanoEncoder`` already has, so it's a drop-in replacement for that
encoder in ``baselines/reciprocal_lambda_grid/eval.py``'s scoring code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

from baselines.voyage_nano.encode import l2_normalize
from twotower.eval import encode_role, load_model_for_eval

TOP1_CTRL_ADAPTER_DIR = Path("artifacts/twotower_top1_optimised/top1_ctrl_001/adapter")
TOP1_CTRL_BASE_MODEL = "voyageai/voyage-4-nano"
TOP1_CTRL_MAX_SEQ_LENGTH = 4096  # matches twotower/config.py's TrainConfig default
TOP1_CTRL_TRUNCATE_DIM = 1024


class Top1CtrlEncoder:
    """Loads voyage-4-nano + the top1_ctrl LoRA adapter once; caches encodes to disk."""

    def __init__(
        self,
        *,
        adapter_dir: Path = TOP1_CTRL_ADAPTER_DIR,
        model_name: str = TOP1_CTRL_BASE_MODEL,
        device: str,
        max_seq_length: int = TOP1_CTRL_MAX_SEQ_LENGTH,
        truncate_dim: int = TOP1_CTRL_TRUNCATE_DIM,
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
            dim = self.model.get_sentence_embedding_dimension() or TOP1_CTRL_TRUNCATE_DIM
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
