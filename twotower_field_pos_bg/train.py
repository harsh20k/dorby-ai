"""LoRA fine-tune, `top1_ctrl`'s exact recipe, on positioning+background-only
text (both seeker and candidate).

Near-copy of `twotower_top1_optimised/train.py`'s control corner (plain
`MultipleNegativesRankingLoss(scale=20.0)`, no hardness weighting) with the
hardness-weighting branch removed entirely — this package never uses it.
Reuses generic, model-agnostic helpers from `twotower.train`/`twotower.eval`
read-only, exactly as every other package in this project does.

Usage:
  python -m twotower_field_pos_bg.train --run-id field_pos_bg_001 \\
      --rows-path artifacts/twotower_field_pos_bg/rrf_003_multineg_k1_pos_bg.json \\
      --dry-run
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

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
    build_model,
    collect_env_metadata,
    select_best_checkpoint,
    smoke_backward,
    truncation_stats,
)
from twotower_field_pos_bg.config import build_config
from twotower_field_pos_bg.data import carve_dev, load_multineg_rows, rows_to_hf_dict
from twotower_field_pos_bg.eval_dev import CorpusRecallDevEvaluator


def run_training(
    cfg: TrainConfig,
    rows_path: Path,
    *,
    negatives_per_anchor: int = 1,
    dry_run: bool = False,
    run_holdout: bool = False,
    real_holdout_data_dir: Path = Path("data"),
    real_holdout_split_path: Path = Path("data/synthetic/seed_split.json"),
    dev_seeker_fraction: float = 0.1,
    dev_min_rows: int = 20,
    resume_from_checkpoint: str | None = None,
) -> dict[str, Any]:
    device = pick_device()
    if device == "mps":
        print("warning: MPS selected — prefer Modal CUDA (twotower_field_pos_bg/modal_train.py)")

    output_dir = Path(cfg.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not resume_from_checkpoint:
        raise FileExistsError(f"output_dir {output_dir} is non-empty; pass a fresh run id or resume")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows, extraction_summary = load_multineg_rows(rows_path)
    train_rows, dev_rows = carve_dev(
        all_rows, seeker_fraction=dev_seeker_fraction, min_rows=dev_min_rows, seed=cfg.seed
    )
    print(f"rows: train={len(train_rows)} dev={len(dev_rows)} (source: {extraction_summary})")

    model = build_model(cfg, device)
    lora_info = add_lora_adapter(model, cfg)
    print(f"lora: {lora_info}")

    all_texts = (
        [r.anchor for r in train_rows]
        + [r.positive for r in train_rows]
        + [n for r in train_rows for n in r.negatives]
    )
    trunc = truncation_stats(model, all_texts, cfg.max_seq_length)
    print(f"truncation: {trunc}")

    smoke_backward(model, cfg)
    print("smoke_backward: ok")

    neg_columns = [f"negative_{i + 1}" for i in range(negatives_per_anchor)]
    train_ds = Dataset.from_dict(rows_to_hf_dict(train_rows, negatives_per_anchor=negatives_per_anchor))
    train_ds = train_ds.select_columns(["anchor", "positive", *neg_columns])

    bf16, fp16 = resolve_mixed_precision(device if device == "cuda" else "cpu", cfg)
    prompts = {"anchor": cfg.query_prompt, "positive": cfg.document_prompt}
    prompts.update({col: cfg.document_prompt for col in neg_columns})

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
        prompts=prompts,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        report_to=[],
        run_name=output_dir.name,
    )

    loss_scale = float(cfg.extra["loss_scale"])
    loss = MultipleNegativesRankingLoss(model=model, scale=loss_scale)
    print(f"loss: MultipleNegativesRankingLoss(scale={loss_scale})")

    evaluator = CorpusRecallDevEvaluator(dev_rows, batch_size=cfg.eval_batch_size, name="train_dev")
    print(f"dev corpus: {len(evaluator.corpus_ids)} unique candidates over {len(dev_rows)} rows")

    padded_rows = sum(1 for r in train_rows + dev_rows if r.padded_count > 0)
    total_rows = len(train_rows) + len(dev_rows)
    meta = collect_env_metadata(cfg, device)
    meta.update(
        {
            "experiment": "twotower_field_pos_bg",
            "seeker_fields": ["positioning", "background"],
            "candidate_fields": ["positioning", "background"],
            "rows_path": str(rows_path),
            "negatives_per_anchor": negatives_per_anchor,
            "extraction_summary": extraction_summary,
            "n_train_rows": len(train_rows),
            "n_dev_rows": len(dev_rows),
            "rows_with_any_padding": padded_rows,
            "total_rows": total_rows,
            "padding_rate": padded_rows / total_rows if total_rows else None,
            "lora": lora_info,
            "truncation": trunc,
            "loss": "MultipleNegativesRankingLoss",
            "loss_scale": loss_scale,
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
        json.dumps({"log_history": log_history, "train_loss": train_loss_points, "eval": eval_points}, indent=2) + "\n"
    )
    print(
        f"loss_history: {len(train_loss_points)} logging points, "
        f"first_loss={train_loss_points[0]['loss'] if train_loss_points else None}, "
        f"last_loss={train_loss_points[-1]['loss'] if train_loss_points else None}"
    )

    dev_cfg = dataclasses.replace(cfg, primary_metric=cfg.primary_metric)
    model, best_info = select_best_checkpoint(
        model, checkpoints_dir=output_dir / "checkpoints", cfg=dev_cfg, device=device
    )
    print(f"selected checkpoint: {best_info}")

    final_dir = output_dir / "adapter"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))

    dev_evaluator_final = CorpusRecallDevEvaluator(dev_rows, batch_size=cfg.eval_batch_size, name="final_dev")
    dev_metrics = dev_evaluator_final(model)
    (output_dir / "metrics_train_dev.json").write_text(json.dumps(dev_metrics, indent=2) + "\n")

    # No `--run-holdout` path here: the real holdout pairs carry full-profile
    # text, not the trimmed positioning+background this model was trained on
    # (twotower.eval.evaluate_pairs has no notion of field trimming), so
    # scoring against it here would silently feed the model an out-of-
    # distribution input. Real-pair scoring for this experiment goes through
    # `twotower_field_pos_bg.eval.run_eval` instead (all 200, trimmed both
    # sides) — see modal_eval.py.
    result = {
        **meta,
        "adapter_dir": str(final_dir),
        "loss_history_path": str(output_dir / "loss_history.json"),
        "final_train_loss": train_loss_points[-1]["loss"] if train_loss_points else None,
        "best_checkpoint": best_info,
        "metrics_train_dev": dev_metrics,
    }
    (output_dir / "run_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"saved adapter -> {final_dir}")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--rows-path", type=Path, required=True)
    p.add_argument("--negatives-per-anchor", type=int, default=1)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-batch-size", type=int, default=6)
    p.add_argument("--eval-batch-size", type=int, default=6)
    p.add_argument("--gradient-accumulation-steps", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--loss-scale", type=float, default=20.0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume-from-checkpoint", type=str, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = build_config(
        "voyage-4-nano",
        run_id=args.run_id,
        rows_path=args.rows_path,
        negatives_per_anchor=args.negatives_per_anchor,
        epochs=args.epochs,
        seed=args.seed,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        loss_scale=args.loss_scale,
    )
    result = run_training(
        cfg,
        args.rows_path,
        negatives_per_anchor=args.negatives_per_anchor,
        dry_run=args.dry_run,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "config"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
