"""LoRA fine-tune with KL-regularized MultipleNegativesRankingLoss —
`voyage_gemini_ctrl_001`'s exact recipe and data, plus `twotower_kl_reg`'s
KL-leash-to-frozen-base loss term.

Custom training loop (not `SentenceTransformerTrainer`'s stock loss
interface), copied from `twotower_kl_reg/train.py`, because the KL penalty
needs a second forward pass through the same PEFT model with its adapter
toggled off (`disable_adapters()`/`enable_adapters()`). Reuses generic,
model-agnostic helpers from `twotower.train`/`twotower.eval` read-only; copies
(does not import) `twotower_voyage_gemini_ctrl`'s `data.py`/`eval_dev.py` and
`twotower_kl_reg`'s `losses.py`, per the isolation rule. See this package's
`__init__.py` for the full rationale.

Usage:
  python -m twotower_voyage_gemini_kl.train --run-id local_smoke \\
      --rows-path artifacts/twotower_voyage_gemini_ctrl/voyage_gemini_smoke002_multineg_k1.json \\
      --kl-weight 0.5 --dry-run --no-run-holdout
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
    from sentence_transformers.sentence_transformer.training_args import BatchSamplers
except ImportError:  # ST 3.x
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
    truncation_stats,
)
from twotower_voyage_gemini_kl.config import build_config
from twotower_voyage_gemini_kl.data import carve_dev, load_multineg_rows, rows_to_hf_dict
from twotower_voyage_gemini_kl.eval_dev import CorpusRecallDevEvaluator
from twotower_voyage_gemini_kl.losses import KLRegularizedMNRL


def smoke_kl_backward(model, cfg: TrainConfig) -> dict[str, Any]:
    """Verify the KL mechanism before any GPU spend.

    (1) At LoRA init the B matrix is zero, so the adapter-enabled and
    adapter-disabled forward passes must be numerically identical — if they
    are not, `disable_adapters()` is not doing what this loss assumes.
    (2) Backward through the combined loss must produce non-zero LoRA
    gradients — the same check every prior experiment in this project runs
    (`twotower.train.smoke_backward`), extended to the new loss.
    (3) The adapter must be left enabled afterward (the toggle inside
    `forward()` must not leak a disabled state into the rest of training).
    """
    model.train()
    device = next(model.parameters()).device
    preprocess = getattr(model, "preprocess", None) or model.tokenize

    def feats(texts: list[str]) -> dict:
        raw = dict(preprocess(texts))
        return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in raw.items()}

    sentence_features = [
        feats(["smoke query about fundraising", "smoke query about hiring"]),
        feats(["smoke document about venture partners", "smoke document about recruiting"]),
        feats(["smoke document about unrelated topic", "smoke document about a different field"]),
    ]

    loss_fn = KLRegularizedMNRL(
        model, scale=float(cfg.extra["loss_scale"]), kl_weight=float(cfg.extra["kl_weight"])
    )

    with torch.no_grad():
        adapted0 = loss_fn._embed(sentence_features)
        frozen0 = loss_fn._frozen_embed(sentence_features)
    max_diff_at_init = max(float((a - f).detach().abs().max()) for a, f in zip(adapted0, frozen0))
    if max_diff_at_init > 1e-4:
        raise RuntimeError(
            f"smoke_kl_backward: adapted != frozen at LoRA init (max abs diff "
            f"{max_diff_at_init:.6f}) — LoRA's B matrix is not zero-initialized as "
            f"expected, or disable_adapters() is not producing the true frozen output."
        )

    model.zero_grad(set_to_none=True)
    loss = loss_fn(sentence_features, labels=None)
    loss.backward()

    lora_grad_sum = 0.0
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None and "lora_" in name.lower():
            lora_grad_sum += float(p.grad.detach().abs().sum())
    if lora_grad_sum <= 0:
        raise RuntimeError("smoke_kl_backward: no non-zero LoRA gradients after backward")

    auto = model[0].auto_model
    active_after = auto.active_adapters() if hasattr(auto, "active_adapters") else None

    return {
        "max_diff_at_init": max_diff_at_init,
        "lora_grad_sum": lora_grad_sum,
        "smoke_loss": float(loss.detach()),
        "active_adapters_after": active_after,
    }


def run_training(
    cfg: TrainConfig,
    rows_path: Path,
    *,
    negatives_per_anchor: int = 1,
    dry_run: bool = False,
    run_holdout: bool = True,
    real_holdout_data_dir: Path = Path("data"),
    real_holdout_split_path: Path = Path("data/synthetic/seed_split.json"),
    dev_seeker_fraction: float = 0.1,
    dev_min_rows: int = 20,
    resume_from_checkpoint: str | None = None,
) -> dict[str, Any]:
    device = pick_device()
    if device == "mps":
        print("warning: MPS selected — prefer Modal CUDA (twotower_voyage_gemini_kl/modal_train.py)")

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

    smoke_info = smoke_kl_backward(model, cfg)
    print(f"smoke_kl_backward: ok {smoke_info}")

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
    kl_weight = float(cfg.extra["kl_weight"])
    loss = KLRegularizedMNRL(model, scale=loss_scale, kl_weight=kl_weight)
    print(f"loss: KLRegularizedMNRL(scale={loss_scale}, kl_weight={kl_weight})")

    evaluator = CorpusRecallDevEvaluator(dev_rows, batch_size=cfg.eval_batch_size, name="train_dev")
    print(f"dev corpus: {len(evaluator.corpus_ids)} unique candidates over {len(dev_rows)} rows")

    padded_rows = sum(1 for r in train_rows + dev_rows if r.padded_count > 0)
    total_rows = len(train_rows) + len(dev_rows)
    meta = collect_env_metadata(cfg, device)
    meta.update(
        {
            "experiment": "twotower_voyage_gemini_kl",
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
            "loss": "KLRegularizedMNRL",
            "loss_scale": loss_scale,
            "kl_weight": kl_weight,
            "smoke_kl_backward": smoke_info,
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
                "retrieval_recall_at_1": holdout_metrics["retrieval"].get("recall@1"),
                "retrieval_recall_at_10": holdout_metrics["retrieval"].get("recall@10"),
                "hard_negative_auc": holdout_metrics["slices"].get("neg_hardness", {}).get("hard", {}).get("pair_auc"),
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
    p.add_argument("--run-id", required=True)
    p.add_argument(
        "--rows-path",
        type=Path,
        default=Path("artifacts/twotower_voyage_gemini_ctrl/voyage_gemini_smoke002_multineg_k1.json"),
    )
    p.add_argument("--negatives-per-anchor", type=int, default=1)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-batch-size", type=int, default=6)
    p.add_argument("--eval-batch-size", type=int, default=6)
    p.add_argument("--gradient-accumulation-steps", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--loss-scale", type=float, default=20.0)
    p.add_argument("--kl-weight", type=float, default=0.5)
    p.add_argument("--run-holdout", action="store_true", default=True)
    p.add_argument("--no-run-holdout", dest="run_holdout", action="store_false")
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
        kl_weight=args.kl_weight,
    )
    result = run_training(
        cfg,
        args.rows_path,
        negatives_per_anchor=args.negatives_per_anchor,
        dry_run=args.dry_run,
        run_holdout=args.run_holdout,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "config"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
