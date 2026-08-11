"""Disk-cached encoder wrapper around the fine-tuned ``voyage_gemini_ctrl`` checkpoint.

Deliberate duplicate of ``baselines/reciprocal_lambda_grid_top1ctrl/encode.py``
(experiment isolation rule — each fine-tuned-model sweep gets its own package
so its numbers stay reproducible independently).

``voyage_gemini_ctrl_001`` (``twotower_voyage_gemini_ctrl``) is
``top1_ctrl_001``'s exact recipe (LoRA rank 8, q/k/v/o_proj, same loss/lr/
epochs/text builders) retrained on a bigger, newer synthetic batch
(``pairing_voyage_gemini``'s ``smoke_test_002``, 3,008 rows vs. ``top1_ctrl``'s
643) — see `docs/twotower-voyage-gemini-ctrl-experiment.md`. Holdout pair AUC
0.6802 (vs. `top1_ctrl`'s 0.5974) — currently the project's best fine-tuned
checkpoint on this metric, though that batch's own leakage checks found it
measurably leakier than `rrf_003` (candidate-only AUC 0.758 vs. 0.634), a
caveat the linked doc keeps up front.

Reuses ``twotower/eval.py::load_model_for_eval``/``encode_role`` unmodified
and read-only, same as the top1_ctrl wrapper — only the adapter path and the
disk-cache-by-``cache_name`` wrapper are this package's own.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

from baselines.voyage_nano.encode import l2_normalize
from twotower.eval import encode_role, load_model_for_eval

VOYAGE_GEMINI_CTRL_ADAPTER_DIR = Path("artifacts/twotower_voyage_gemini_ctrl/voyage_gemini_ctrl_001/adapter")
VOYAGE_GEMINI_CTRL_BASE_MODEL = "voyageai/voyage-4-nano"
VOYAGE_GEMINI_CTRL_MAX_SEQ_LENGTH = 4096  # matches twotower/config.py's TrainConfig default
VOYAGE_GEMINI_CTRL_TRUNCATE_DIM = 1024


class VoyageGeminiCtrlEncoder:
    """Loads voyage-4-nano + the voyage_gemini_ctrl LoRA adapter once; caches encodes to disk."""

    def __init__(
        self,
        *,
        adapter_dir: Path = VOYAGE_GEMINI_CTRL_ADAPTER_DIR,
        model_name: str = VOYAGE_GEMINI_CTRL_BASE_MODEL,
        device: str,
        max_seq_length: int = VOYAGE_GEMINI_CTRL_MAX_SEQ_LENGTH,
        truncate_dim: int = VOYAGE_GEMINI_CTRL_TRUNCATE_DIM,
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
            dim = self.model.get_sentence_embedding_dimension() or VOYAGE_GEMINI_CTRL_TRUNCATE_DIM
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
