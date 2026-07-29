"""Mid-training dev evaluator for the triplet fine-tune (checkpoint-selection
signal only — the number that actually matters is the final real-holdout
eval in twotower.eval, run separately after training).

Writes the same file layout twotower.train.select_best_checkpoint already
knows how to read (train_dev_metrics_epoch{E}_steps{S}.json under
checkpoints/eval/, with a "flat" dict keyed by "train_dev_<primary_metric>")
so that shared, unmodified selection logic works unchanged on this
experiment's checkpoints too.
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
from twotower_rrf_triplet.data import Triplet


class TripletDevEvaluator(SentenceEvaluator):
    """Fraction of dev triplets where cos(anchor, positive) > cos(anchor, negative)."""

    def __init__(
        self,
        triplets: Sequence[Triplet],
        *,
        batch_size: int = 4,
        name: str = "train_dev",
    ) -> None:
        super().__init__()
        self.triplets = list(triplets)
        self.batch_size = batch_size
        self.name = name
        self.primary_metric = f"{name}_triplet_accuracy"
        self.greater_is_better = True

    def __call__(
        self,
        model: SentenceTransformer,
        output_path: str | None = None,
        epoch: int = -1,
        steps: int = -1,
    ) -> dict[str, float]:
        anchors = [t.anchor for t in self.triplets]
        positives = [t.positive for t in self.triplets]
        negatives = [t.negative for t in self.triplets]

        anchor_emb = l2_normalize(
            np.asarray(
                model.encode_query(
                    anchors,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                ),
                dtype=np.float32,
            )
        )
        pos_emb = l2_normalize(
            np.asarray(
                model.encode_document(
                    positives,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                ),
                dtype=np.float32,
            )
        )
        neg_emb = l2_normalize(
            np.asarray(
                model.encode_document(
                    negatives,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                ),
                dtype=np.float32,
            )
        )

        pos_scores = cosine_scores(anchor_emb, pos_emb)
        neg_scores = cosine_scores(anchor_emb, neg_emb)
        accuracy = float(np.mean(pos_scores > neg_scores))
        margin = float(np.mean(pos_scores - neg_scores))

        flat = {f"{self.name}_triplet_accuracy": accuracy, f"{self.name}_margin": margin}
        raw = {
            "n_triplets": len(self.triplets),
            "accuracy": accuracy,
            "mean_margin": margin,
            "mean_pos_score": float(np.mean(pos_scores)),
            "mean_neg_score": float(np.mean(neg_scores)),
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
