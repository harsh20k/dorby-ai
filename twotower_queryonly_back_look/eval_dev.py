"""Dev evaluator that ranks against a real corpus, so checkpoint selection can see recall@1.

Copy of ``twotower_top1_optimised/eval_dev.py`` (pinned by
``tests/test_queryonly_back_look.py``, import path updated only) — reused unchanged
so checkpoint selection works identically to `top1_ctrl`'s and any
difference in final results is attributable to the field trimming, not a
different selection rule.

Scoring is delegated to ``baselines.metrics.retrieval_metrics`` **unchanged**
— the same function ``twotower.eval.evaluate_pairs`` calls for the real
holdout.
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

from baselines.metrics import retrieval_metrics
from baselines.voyage_nano.encode import l2_normalize

from twotower_queryonly_back_look.data import MultiNegRow


def build_dev_corpus(rows: Sequence[MultiNegRow]) -> tuple[list[str], list[str]]:
    """Unique (candidate_id, text) across every positive and negative in the split.

    Positives are inserted before negatives so that a candidate appearing as both
    keeps its positive-side text, matching `twotower.eval.build_candidate_corpus`.
    """
    ids: list[str] = []
    texts: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if row.positive_id not in seen:
            seen.add(row.positive_id)
            ids.append(row.positive_id)
            texts.append(row.positive)
    for row in rows:
        for cid, text in zip(row.negative_ids, row.negatives):
            if cid in seen:
                continue
            seen.add(cid)
            ids.append(cid)
            texts.append(text)
    return ids, texts


class CorpusRecallDevEvaluator(SentenceEvaluator):
    """Ranks dev anchors against the full dev corpus; primary metric is recall@1."""

    def __init__(
        self,
        rows: Sequence[MultiNegRow],
        *,
        batch_size: int = 8,
        name: str = "train_dev",
        primary_metric: str = "recall@1",
    ) -> None:
        super().__init__()
        self.rows = list(rows)
        self.batch_size = batch_size
        self.name = name
        self.primary_metric = f"{name}_{primary_metric}"
        self.greater_is_better = True
        self.corpus_ids, self.corpus_texts = build_dev_corpus(self.rows)

    def _encode(self, model: SentenceTransformer, texts: Sequence[str], role: str) -> np.ndarray:
        fn = model.encode_query if role == "query" else model.encode_document
        raw = fn(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return l2_normalize(np.asarray(raw, dtype=np.float32))

    def __call__(
        self,
        model: SentenceTransformer,
        output_path: str | None = None,
        epoch: int = -1,
        steps: int = -1,
    ) -> dict[str, float]:
        anchor_emb = self._encode(model, [r.anchor for r in self.rows], "query")
        corpus_emb = self._encode(model, self.corpus_texts, "document")

        metrics = retrieval_metrics(
            query_embs=anchor_emb,
            target_ids=[r.positive_id for r in self.rows],
            candidate_ids=self.corpus_ids,
            candidate_embs=corpus_emb,
        )

        flat = {
            f"{self.name}_recall@1": float(metrics.get("recall@1", 0.0)),
            f"{self.name}_recall@5": float(metrics.get("recall@5", 0.0)),
            f"{self.name}_recall@10": float(metrics.get("recall@10", 0.0)),
            f"{self.name}_mrr": float(metrics.get("mrr", 0.0)),
            f"{self.name}_mean_rank": float(metrics.get("mean_rank", 0.0)),
        }
        raw: dict[str, Any] = {
            "n_rows": len(self.rows),
            "n_corpus": len(self.corpus_ids),
            "retrieval": metrics,
        }
        print(
            f"  dev[{self.name}] corpus={len(self.corpus_ids)} "
            f"R@1={flat[f'{self.name}_recall@1']:.4f} "
            f"R@10={flat[f'{self.name}_recall@10']:.4f} "
            f"MRR={flat[f'{self.name}_mrr']:.4f}"
        )

        if output_path is not None:
            out = Path(output_path)
            out.mkdir(parents=True, exist_ok=True)
            suffix = f"_epoch{epoch}" if epoch != -1 else ""
            if steps != -1:
                suffix += f"_steps{steps}"
            (out / f"{self.name}_metrics{suffix}.json").write_text(
                json.dumps({"raw": raw, "flat": flat}, indent=2) + "\n"
            )
        return flat
