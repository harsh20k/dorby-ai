"""LoRA fine-tune with MultipleNegativesRankingLoss on rrf_003 triplets.

Separate experiment from twotower/'s Arm A/B/C (real-data, pairwise
ContrastiveLoss) pipeline — see docs/twotower-rrf-triplet-experiment.md for
why, and for the caveat that rrf_003's labels are an LLM judge's opinion on
synthetic profiles, not real accept/decline outcomes. Reuses generic,
model-agnostic helpers from twotower.train/twotower.eval read-only; does not
import or modify anything under twotower/'s Arm A/B/C output paths.

Usage:
  python -m twotower_rrf_triplet.train --preset voyage-4-nano --run-id rrf_triplet_voyage_nano_001 --dry-run
  python -m twotower_rrf_triplet.train --preset qwen3-8b --run-id rrf_triplet_qwen3_8b_001 --dry-run
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
try:
    from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
    from sentence_transformers.sentence_transformer.training_args import BatchSamplers
except ImportError:  # ST 3.x
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers

from baselines.voyage_nano.encode import pick_device
from twotower.config import TrainConfig, resolve_mixed_precision
from twotower.data import assert_no_holdout_leak, build_split_bundle
from twotower.eval import evaluate_pairs
from twotower.train import (
    add_lora_adapter,
    collect_env_metadata,
    smoke_backward,
    truncation_stats,
)
from twotower_rrf_triplet.checkpoint import select_best_checkpoint_with_dtype
from twotower_rrf_triplet.config import build_config
from twotower_rrf_triplet.data import Triplet, carve_dev, load_triplets, triplets_to_hf_dict
from twotower_rrf_triplet.eval_dev import TripletDevEvaluator
from twotower_rrf_triplet.model import build_model_with_dtype

_TORCH_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def run_training(
    cfg: TrainConfig,
    triplets_path: Path,
    *,
    dry_run: bool = False,
    run_holdout: bool = True,
    real_holdout_data_dir: Path = Path("data"),
    real_holdout_split_path: Path = Path("data/synthetic/seed_split.json"),
    dev_seeker_fraction: float = 0.1,
    dev_min_triplets: int = 20,
    resume_from_checkpoint: str | None = None,
) -> dict[str, Any]:
    device = pick_device()
    if device == "mps":
        print("warning: MPS selected — prefer Modal CUDA (twotower_rrf_triplet/modal_train.py)")

    output_dir = Path(cfg.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not resume_from_checkpoint:
        raise FileExistsError(f"output_dir {output_dir} is non-empty; pass a fresh run id or resume")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_triplets, extraction_summary = load_triplets(triplets_path)
    train_rows, dev_rows = carve_dev(
        all_triplets, seeker_fraction=dev_seeker_fraction, min_triplets=dev_min_triplets, seed=cfg.seed
    )
    print(f"triplets: train={len(train_rows)} dev={len(dev_rows)} (source: {extraction_summary})")

    torch_dtype = _TORCH_DTYPES.get(cfg.extra.get("torch_dtype"))
    model = build_model_with_dtype(
        cfg,
        device,
        torch_dtype=torch_dtype,
        gradient_checkpointing=bool(cfg.extra.get("gradient_checkpointing_override")),
    )
    lora_info = add_lora_adapter(model, cfg)
    print(f"lora: {lora_info}")

    all_texts = (
        [t.anchor for t in train_rows] + [t.positive for t in train_rows] + [t.negative for t in train_rows]
    )
    trunc = truncation_stats(model, all_texts, cfg.max_seq_length)
    print(f"truncation: {trunc}")

    smoke_backward(model, cfg)
    print("smoke_backward: ok")

    train_ds = Dataset.from_dict(triplets_to_hf_dict(train_rows))
    train_ds = train_ds.select_columns(["anchor", "positive", "negative"])

    bf16, fp16 = resolve_mixed_precision(device if device == "cuda" else "cpu", cfg)
    args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        max_grad_norm=cfg.max_grad_norm,
        fp16=fp16,
        bf16=bf16,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=False,
        logging_steps=cfg.logging_steps,
        seed=cfg.seed,
        data_seed=cfg.seed,
        prompts={
            "anchor": cfg.query_prompt,
            "positive": cfg.document_prompt,
            "negative": cfg.document_prompt,
        },
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        report_to=[],
        run_name=output_dir.name,
    )

    loss = MultipleNegativesRankingLoss(model=model)
    evaluator = TripletDevEvaluator(dev_rows, batch_size=cfg.eval_batch_size, name="train_dev")

    meta = collect_env_metadata(cfg, device)
    meta.update(
        {
            "experiment": "twotower_rrf_triplet",
            "triplets_path": str(triplets_path),
            "extraction_summary": extraction_summary,
            "n_train_triplets": len(train_rows),
            "n_dev_triplets": len(dev_rows),
            "lora": lora_info,
            "truncation": trunc,
            "loss": "MultipleNegativesRankingLoss",
            "dry_run": dry_run,
        }
    )
    (output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    if dry_run:
        print("dry_run: skipping trainer.train()")
        return meta

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        loss=loss,
        evaluator=evaluator,
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    log_history = list(trainer.state.log_history)
    train_loss_points = [
        {"step": e["step"], "epoch": e.get("epoch"), "loss": e["loss"]}
        for e in log_history
        if "loss" in e and "step" in e
    ]
    eval_points = [
        {"step": e.get("step"), "epoch": e.get("epoch"), **{k: v for k, v in e.items() if k.startswith("eval_")}}
        for e in log_history
        if any(k.startswith("eval_") for k in e)
    ]
    (output_dir / "loss_history.json").write_text(
        json.dumps(
            {"log_history": log_history, "train_loss": train_loss_points, "eval": eval_points},
            indent=2,
        )
        + "\n"
    )
    print(
        f"loss_history: {len(train_loss_points)} logging points, "
        f"first_loss={train_loss_points[0]['loss'] if train_loss_points else None}, "
        f"last_loss={train_loss_points[-1]['loss'] if train_loss_points else None}"
    )

    dev_cfg = dataclasses.replace(cfg, primary_metric="triplet_accuracy")
    model, best_info = select_best_checkpoint_with_dtype(
        model,
        checkpoints_dir=output_dir / "checkpoints",
        cfg=dev_cfg,
        device=device,
        torch_dtype=torch_dtype,
        gradient_checkpointing=bool(cfg.extra.get("gradient_checkpointing_override")),
    )
    print(f"selected checkpoint: {best_info}")

    final_dir = output_dir / "adapter"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))

    dev_evaluator_final = TripletDevEvaluator(dev_rows, batch_size=cfg.eval_batch_size, name="final_dev")
    dev_metrics = dev_evaluator_final(model)
    (output_dir / "metrics_train_dev.json").write_text(json.dumps(dev_metrics, indent=2) + "\n")

    holdout_metrics = None
    if run_holdout:
        bundle = build_split_bundle(real_holdout_data_dir, real_holdout_split_path)
        assert_no_holdout_leak(bundle, split_path=real_holdout_split_path)
        holdout_metrics = evaluate_pairs(
            model,
            bundle.holdout,
            batch_size=cfg.eval_batch_size,
            model_name=cfg.model_name,
            device=device,
            max_length=cfg.max_seq_length,
            truncate_dim=cfg.truncate_dim,
        )
        (output_dir / "metrics_holdout.json").write_text(json.dumps(holdout_metrics, indent=2) + "\n")

    result = {
        **meta,
        "adapter_dir": str(final_dir),
        "loss_history_path": str(output_dir / "loss_history.json"),
        "final_train_loss": train_loss_points[-1]["loss"] if train_loss_points else None,
        "best_checkpoint": best_info,
        "metrics_train_dev": dev_metrics,
        "metrics_holdout": (
            {
                "pair_auc": holdout_metrics["pair"]["roc_auc"],
                "pair_ap": holdout_metrics["pair"]["average_precision"],
                "retrieval_mrr": holdout_metrics["retrieval"]["mrr"],
                "hard_negative_auc": holdout_metrics["slices"]
                .get("neg_hardness", {})
                .get("hard", {})
                .get("pair_auc"),
            }
            if holdout_metrics
            else None
        ),
    }
    (output_dir / "run_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"saved adapter -> {final_dir}")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=("voyage-4-nano", "qwen3-8b"), required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument(
        "--triplets-path",
        type=Path,
        default=Path("artifacts/twotower_rrf_triplet/rrf_003_triplets.json"),
    )
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-holdout", action="store_true", default=True)
    p.add_argument("--no-run-holdout", dest="run_holdout", action="store_false")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume-from-checkpoint", type=str, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = build_config(
        args.preset,
        run_id=args.run_id,
        triplets_path=args.triplets_path,
        epochs=args.epochs,
        seed=args.seed,
    )
    result = run_training(
        cfg,
        args.triplets_path,
        dry_run=args.dry_run,
        run_holdout=args.run_holdout,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "config"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
