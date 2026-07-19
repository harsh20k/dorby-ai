"""Holdout / train-dev evaluation for LoRA two-tower models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer
try:
    from sentence_transformers.sentence_transformer.evaluation import SentenceEvaluator
except ImportError:  # ST 3.x
    from sentence_transformers.evaluation import SentenceEvaluator

from baselines.metrics import pair_metrics, print_metrics, retrieval_metrics, slice_metrics
from baselines.voyage_nano.encode import cosine_scores, l2_normalize, pick_device
from twotower.config import TrainConfig
from twotower.data import LabeledPair, SplitBundle, assert_no_holdout_leak, build_split_bundle


def build_candidate_corpus(
    pairs: Sequence[LabeledPair],
) -> tuple[list[str], list[str]]:
    """Unique matchContactIds with first-seen candidate text (pos before neg)."""
    ordered = sorted(pairs, key=lambda p: (0 if p.label == "pos" else 1, p.pair_id))
    ids: list[str] = []
    texts: list[str] = []
    seen: set[str] = set()
    for item in ordered:
        mid = item.pair["matchContactId"]
        if mid in seen:
            continue
        seen.add(mid)
        ids.append(mid)
        texts.append(item.candidate_text)
    return ids, texts


def load_model_for_eval(
    *,
    model_name: str,
    adapter_dir: Path | None,
    device: str,
    max_seq_length: int,
    truncate_dim: int,
    trust_remote_code: bool = True,
    model_revision: str | None = None,
) -> SentenceTransformer:
    model = SentenceTransformer(
        model_name,
        trust_remote_code=trust_remote_code,
        truncate_dim=truncate_dim,
        device=device,
        revision=model_revision,
    )
    model.max_seq_length = max_seq_length
    if adapter_dir is not None:
        adapter_dir = Path(adapter_dir)
        # Prefer adapter saved via PEFT / ST save_pretrained.
        if (adapter_dir / "adapter_config.json").exists() or (
            adapter_dir / "adapter_model.safetensors"
        ).exists():
            model.load_adapter(str(adapter_dir))
        elif adapter_dir.exists():
            # Full ST checkpoint directory.
            model = SentenceTransformer(
                str(adapter_dir),
                trust_remote_code=trust_remote_code,
                truncate_dim=truncate_dim,
                device=device,
            )
            model.max_seq_length = max_seq_length
        else:
            raise FileNotFoundError(f"adapter/checkpoint not found: {adapter_dir}")
    return model


def encode_role(
    model: SentenceTransformer,
    texts: Sequence[str],
    *,
    role: Literal["query", "document"],
    batch_size: int,
) -> np.ndarray:
    encode_fn = model.encode_query if role == "query" else model.encode_document
    raw = encode_fn(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return l2_normalize(np.asarray(raw, dtype=np.float32))


def evaluate_pairs(
    model: SentenceTransformer,
    pairs: Sequence[LabeledPair],
    *,
    batch_size: int = 4,
    model_name: str = "voyageai/voyage-4-nano",
    device: str | None = None,
    max_length: int = 4096,
    truncate_dim: int = 1024,
) -> dict[str, Any]:
    positives = [p for p in pairs if p.label == "pos"]
    negatives = [p for p in pairs if p.label == "neg"]
    if not positives or not negatives:
        raise ValueError(
            f"need both pos and neg pairs for eval; got {len(positives)} pos, "
            f"{len(negatives)} neg"
        )

    pos_seeker = [p.seeker_text for p in positives]
    neg_seeker = [p.seeker_text for p in negatives]
    pos_cand = [p.candidate_text for p in positives]
    neg_cand = [p.candidate_text for p in negatives]
    corpus_ids, corpus_texts = build_candidate_corpus(list(pairs))

    pos_seeker_emb = encode_role(model, pos_seeker, role="query", batch_size=batch_size)
    neg_seeker_emb = encode_role(model, neg_seeker, role="query", batch_size=batch_size)
    pos_cand_emb = encode_role(model, pos_cand, role="document", batch_size=batch_size)
    neg_cand_emb = encode_role(model, neg_cand, role="document", batch_size=batch_size)
    corpus_emb = encode_role(model, corpus_texts, role="document", batch_size=batch_size)

    pos_scores = cosine_scores(pos_seeker_emb, pos_cand_emb)
    neg_scores = cosine_scores(neg_seeker_emb, neg_cand_emb)
    pair = pair_metrics(pos_scores, neg_scores)

    pos_target_ids = [p.pair["matchContactId"] for p in positives]
    retrieval = retrieval_metrics(
        query_embs=pos_seeker_emb,
        target_ids=pos_target_ids,
        candidate_ids=corpus_ids,
        candidate_embs=corpus_emb,
    )
    slices = slice_metrics(
        positives=[p.pair for p in positives],
        negatives=[p.pair for p in negatives],
        pos_scores=pos_scores,
        neg_scores=neg_scores,
        neg_seeker_texts=neg_seeker,
        neg_cand_texts=neg_cand,
        query_embs=pos_seeker_emb,
        target_ids=pos_target_ids,
        candidate_ids=corpus_ids,
        candidate_embs=corpus_emb,
    )
    return {
        "model_name": model_name,
        "device": device or pick_device(),
        "max_length": max_length,
        "truncate_dim": truncate_dim,
        "batch_size": batch_size,
        "n_pos": len(positives),
        "n_neg": len(negatives),
        "pair": pair,
        "retrieval": retrieval,
        "slices": slices,
    }


class PairMetricsEvaluator(SentenceEvaluator):
    """SentenceEvaluator wrapping baselines/metrics.py for train-dev selection."""

    def __init__(
        self,
        pairs: Sequence[LabeledPair],
        *,
        batch_size: int = 4,
        name: str = "train_dev",
        primary_metric: str = "pair_auc",
    ) -> None:
        super().__init__()
        self.pairs = list(pairs)
        self.batch_size = batch_size
        self.name = name
        self.primary_metric = f"{name}_{primary_metric}"
        self.greater_is_better = True

    def __call__(
        self,
        model: SentenceTransformer,
        output_path: str | None = None,
        epoch: int = -1,
        steps: int = -1,
    ) -> dict[str, float]:
        metrics = evaluate_pairs(model, self.pairs, batch_size=self.batch_size)
        flat = {
            "pair_auc": float(metrics["pair"]["roc_auc"]),
            "pair_ap": float(metrics["pair"]["average_precision"]),
            "pair_best_f1": float(metrics["pair"]["best_f1"]),
            "retrieval_mrr": float(metrics["retrieval"]["mrr"]),
            "retrieval_ndcg@10": float(metrics["retrieval"].get("ndcg@10", 0.0)),
            "retrieval_recall@10": float(metrics["retrieval"].get("recall@10", 0.0)),
        }
        prefixed = self.prefix_name_to_metrics(flat, self.name)
        if output_path is not None:
            out = Path(output_path)
            out.mkdir(parents=True, exist_ok=True)
            suffix = f"_epoch{epoch}" if epoch != -1 else ""
            if steps != -1:
                suffix += f"_steps{steps}"
            path = out / f"{self.name}_metrics{suffix}.json"
            path.write_text(json.dumps({"raw": metrics, "flat": prefixed}, indent=2) + "\n")
        return prefixed


def run_eval_cli(
    *,
    data_dir: Path,
    split_path: Path,
    split_name: Literal["holdout", "train_dev", "train"],
    adapter_dir: Path | None,
    model_name: str,
    batch_size: int,
    max_length: int,
    truncate_dim: int,
    artifacts_dir: Path,
    model_revision: str | None = None,
) -> dict[str, Any]:
    device = pick_device()
    bundle = build_split_bundle(data_dir, split_path)
    assert_no_holdout_leak(bundle, split_path=split_path)

    mapping = {
        "holdout": bundle.holdout,
        "train_dev": bundle.train_dev,
        "train": bundle.train,
    }
    pairs = mapping[split_name]
    print(
        f"eval split={split_name} n={len(pairs)} "
        f"(pos={sum(1 for p in pairs if p.label == 'pos')}, "
        f"neg={sum(1 for p in pairs if p.label == 'neg')})"
    )

    model = load_model_for_eval(
        model_name=model_name,
        adapter_dir=adapter_dir,
        device=device,
        max_seq_length=max_length,
        truncate_dim=truncate_dim,
        model_revision=model_revision,
    )
    metrics = evaluate_pairs(
        model,
        pairs,
        batch_size=batch_size,
        model_name=model_name,
        device=device,
        max_length=max_length,
        truncate_dim=truncate_dim,
    )
    metrics["split"] = split_name
    metrics["split_hash"] = bundle.split_hash
    metrics["data_hash"] = bundle.data_hash
    metrics["counts"] = bundle.counts
    if adapter_dir is not None:
        metrics["adapter_dir"] = str(adapter_dir)
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    cfg = TrainConfig()
    p = argparse.ArgumentParser(description="Evaluate LoRA two-tower checkpoint")
    p.add_argument("--data-dir", type=Path, default=cfg.data_dir)
    p.add_argument("--split-path", type=Path, default=cfg.split_path)
    p.add_argument(
        "--split",
        choices=("holdout", "train_dev", "train"),
        default="holdout",
    )
    p.add_argument("--adapter-dir", type=Path, default=None)
    p.add_argument("--model", type=str, default=cfg.model_name)
    p.add_argument("--model-revision", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=cfg.eval_batch_size)
    p.add_argument("--max-length", type=int, default=cfg.max_seq_length)
    p.add_argument("--truncate-dim", type=int, default=cfg.truncate_dim)
    p.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts/twotower/eval"),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    metrics = run_eval_cli(
        data_dir=args.data_dir,
        split_path=args.split_path,
        split_name=args.split,
        adapter_dir=args.adapter_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        truncate_dim=args.truncate_dim,
        artifacts_dir=args.artifacts_dir,
        model_revision=args.model_revision,
    )
    print_metrics(metrics)
    out = args.artifacts_dir / f"metrics_{args.split}.json"
    out.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
