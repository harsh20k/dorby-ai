"""Mid-training dev evaluator for the bigbatch/multi-negative fine-tune
(checkpoint-selection signal only — the number that actually matters is the
final real-holdout eval via twotower.eval, run separately after training).

Writes the same file layout twotower.train.select_best_checkpoint already
knows how to read (train_dev_metrics_epoch{E}_steps{S}.json under
checkpoints/eval/, with a "flat" dict keyed by "train_dev_<primary_metric>")
so that shared, unmodified selection logic works unchanged on this
experiment's checkpoints too — same convention twotower_rrf_triplet/eval_dev.py
established, reimplemented here (not imported) to keep this package free of
any dependency on that other run's code.

Two dev metrics are reported per checkpoint:
  - beat_all_accuracy: fraction of rows where the positive's cosine score
    exceeds *every* one of that row's unique (non-padded) negatives — a
    strict, recall-flavored check ("did it beat the whole small crowd", not
    just one rival).
  - mean_pairwise_accuracy: averaged over unique negatives only, directly
    comparable to rrf_triplet_voyage_nano_001's single-negative dev metric
    (mean of pos_score > neg_score across all real negative comparisons).

Padded (duplicate) negative slots are excluded from both metrics so a row
that only had one real negative doesn't get double-counted against a repeat
of itself.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from sentence_transformers.sentence_transformer.evaluation import SentenceEvaluator
except ImportError:  # ST 3.x
    from sentence_transformers.evaluation import SentenceEvaluator

from baselines.voyage_nano.encode import cosine_scores, l2_normalize
from twotower_rrf_triplet_bigbatch.data import MultiNegRow


class MultiNegTripletDevEvaluator(SentenceEvaluator):
    """beat_all_accuracy + mean_pairwise_accuracy over unique (non-padded)
    negatives per row."""

    def __init__(
        self,
        rows: Sequence[MultiNegRow],
        *,
        batch_size: int = 16,
        name: str = "train_dev",
    ) -> None:
        super().__init__()
        self.rows = list(rows)
        self.batch_size = batch_size
        self.name = name
        self.primary_metric = f"{name}_beat_all_accuracy"
        self.greater_is_better = True

    def __call__(
        self,
        model: SentenceTransformer,
        output_path: str | None = None,
        epoch: int = -1,
        steps: int = -1,
    ) -> dict[str, float]:
        anchors = [r.anchor for r in self.rows]
        positives = [r.positive for r in self.rows]
        # unique (non-padded) negatives only, per row
        unique_neg_lists = [
            list(r.negatives[: min(r.n_unique_negatives, len(r.negatives))]) or [r.negatives[0]]
            for r in self.rows
        ]
        flat_negs: list[str] = [n for lst in unique_neg_lists for n in lst]

        anchor_emb = l2_normalize(
            np.asarray(
                model.encode_query(
                    anchors, batch_size=self.batch_size, show_progress_bar=False,
                    normalize_embeddings=True, convert_to_numpy=True,
                ),
                dtype=np.float32,
            )
        )
        pos_emb = l2_normalize(
            np.asarray(
                model.encode_document(
                    positives, batch_size=self.batch_size, show_progress_bar=False,
                    normalize_embeddings=True, convert_to_numpy=True,
                ),
                dtype=np.float32,
            )
        )
        neg_emb_flat = l2_normalize(
            np.asarray(
                model.encode_document(
                    flat_negs, batch_size=self.batch_size, show_progress_bar=False,
                    normalize_embeddings=True, convert_to_numpy=True,
                ),
                dtype=np.float32,
            )
        )

        pos_scores = cosine_scores(anchor_emb, pos_emb)  # (n_rows,)

        beat_all_flags: list[bool] = []
        pairwise_flags: list[bool] = []
        all_neg_scores: list[float] = []
        offset = 0
        for i, lst in enumerate(unique_neg_lists):
            n = len(lst)
            row_neg_emb = neg_emb_flat[offset : offset + n]
            offset += n
            row_neg_scores = np.atleast_1d(cosine_scores(anchor_emb[i : i + 1], row_neg_emb))
            beat_all_flags.append(bool(np.all(pos_scores[i] > row_neg_scores)))
            pairwise_flags.extend((pos_scores[i] > row_neg_scores).tolist())
            all_neg_scores.extend(row_neg_scores.tolist())

        beat_all_accuracy = float(np.mean(beat_all_flags)) if beat_all_flags else 0.0
        mean_pairwise_accuracy = float(np.mean(pairwise_flags)) if pairwise_flags else 0.0
        margin = float(np.mean(pos_scores) - np.mean(all_neg_scores)) if all_neg_scores else 0.0

        flat = {
            f"{self.name}_beat_all_accuracy": beat_all_accuracy,
            f"{self.name}_mean_pairwise_accuracy": mean_pairwise_accuracy,
            f"{self.name}_margin": margin,
        }
        raw = {
            "n_rows": len(self.rows),
            "beat_all_accuracy": beat_all_accuracy,
            "mean_pairwise_accuracy": mean_pairwise_accuracy,
            "mean_margin": margin,
            "mean_pos_score": float(np.mean(pos_scores)),
        }

        if output_path is not None:
            out = Path(output_path)
            out.mkdir(parents=True, exist_ok=True)
            suffix = f"_epoch{epoch}" if epoch != -1 else ""
            if steps != -1:
                suffix += f"_steps{steps}"
            path = out / f"{self.name}_metrics{suffix}.json"
            path.write_text(json.dumps({"raw": raw, "flat": flat}, indent=2) + "\n")

        return flat
