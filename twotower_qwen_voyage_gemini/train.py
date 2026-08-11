"""LoRA fine-tune with MultipleNegativesRankingLoss —
twotower_qwen_bigbatch's winning micro-batch-6 recipe, retrained on the
pairing_voyage_gemini rows instead of rrf_003.

Separate experiment from twotower_qwen_bigbatch/, twotower_voyage_gemini_ctrl/,
twotower_rrf_triplet/, and twotower_rrf_triplet_bigbatch/ — see this package's
__init__.py for the full rationale and the leakage caveat found on the new
batch before training. Reuses generic, model-agnostic helpers from
twotower.train/twotower.eval read-only; imports nothing from
twotower_qwen_bigbatch (a prior run's frozen code) except by copy (data.py,
eval_dev.py, model.py, checkpoint.py — pinned byte-identical by
tests/test_qwen_voyage_gemini.py).

Usage (effective batch stays 12, matching twotower_qwen_bigbatch's
qwen_micro6_r1):
  python -m twotower_qwen_voyage_gemini.train --run-id qwen_voyage_gemini_001 \
      --rows-path artifacts/twotower_voyage_gemini_ctrl/voyage_gemini_smoke002_multineg_k1.json \
      --negatives-per-anchor 1 --train-batch-size 6 --gradient-accumulation-steps 2
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any, Callable

from datasets import Dataset
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from transformers import TrainerCallback

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
import torch

from twotower.train import (
    add_lora_adapter,
    collect_env_metadata,
    smoke_backward,
    truncation_stats,
)
from twotower_qwen_voyage_gemini.checkpoint import select_best_checkpoint_with_dtype
from twotower_qwen_voyage_gemini.model import build_model_with_dtype

_TORCH_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
from twotower_qwen_voyage_gemini.config import build_config
from twotower_qwen_voyage_gemini.data import carve_dev, load_multineg_rows, rows_to_hf_dict
from twotower_qwen_voyage_gemini.eval_dev import MultiNegTripletDevEvaluator


def _find_resumable_checkpoint(output_dir: Path) -> str | None:
    """Return the path to the latest valid HF Trainer checkpoint under
    ``output_dir/checkpoints``, or None if none exists yet.

    Added for B200 preemption resilience (see docs/twotower-qwen-voyage-gemini
    -experiment.md): Modal auto-restarts a preempted function "on the same
    input" (https://modal.com/docs/guide/preemption), so a mid-training kill
    re-invokes run_training() with the identical run_id/output_dir. Only a
    directory containing at least one ``checkpoint-<N>/trainer_state.json``
    (the file HF's Trainer writes at the end of each successful save) counts
    as resumable — a directory that only has ``run_meta.json`` (written before
    any checkpoint save; see below) is not, and a directory with unexpected
    contents should still fail loudly rather than being silently adopted.
    """
    ckpt_root = output_dir / "checkpoints"
    if not ckpt_root.exists():
        return None
    candidates = [
        p
        for p in ckpt_root.iterdir()
        if p.is_dir() and p.name.startswith("checkpoint-") and (p / "trainer_state.json").exists()
    ]
    if not candidates:
        return None

    def _step(p: Path) -> int:
        try:
            return int(p.name.rsplit("-", 1)[-1])
        except ValueError:
            return -1

    candidates.sort(key=_step)
    return str(candidates[-1])


def resolve_resume_checkpoint(output_dir: Path, resume_from_checkpoint: str | None) -> str | None:
    """Decide what (if anything) to resume from a possibly-non-empty output_dir.

    Returns the checkpoint path to resume from (or the caller's own explicit
    ``resume_from_checkpoint`` unchanged), or None for a genuine fresh start.
    Raises FileExistsError if the directory is non-empty but contains nothing
    resumable — e.g. only ``run_meta.json`` from a preemption that happened
    before the first HF checkpoint save (``save_strategy="epoch"`` means nothing
    is saved until epoch 1 completes, so a kill earlier than that has no
    checkpoint to resume from and legitimately requires a fresh restart, not a
    silent overwrite).

    Factored out of run_training() so this decision — the actual fix for the
    B200 preemption failures documented in this package's __init__.py — is
    unit-testable without loading the 8B model.
    """
    if not (output_dir.exists() and any(output_dir.iterdir())):
        return None
    if resume_from_checkpoint:
        return resume_from_checkpoint
    detected = _find_resumable_checkpoint(output_dir)
    if detected is not None:
        return detected
    raise FileExistsError(
        f"output_dir {output_dir} is non-empty but no valid HF checkpoint "
        f"(checkpoint-*/trainer_state.json) was found under "
        f"{output_dir / 'checkpoints'} — refusing to silently adopt unknown "
        f"contents. Pass a fresh run id, or --resume-from-checkpoint "
        f"explicitly if you know the right path."
    )


class _CommitOnSaveCallback(TrainerCallback):
    """Fires an arbitrary hook right after each HF Trainer checkpoint save.

    Modal's `Volume.commit()` docstring is explicit that writes are only
    "persisted in durable storage and available to other containers" *after*
    commit() — a mounted volume's writes are not guaranteed durable on their
    own. `train_remote()` in modal_train.py used to call `checkpoints.commit()`
    exactly once, at the very end of training; a preemption between an
    epoch-boundary checkpoint save and that final commit would silently lose
    the checkpoint, defeating resolve_resume_checkpoint()'s whole purpose. This
    callback lets modal_train.py wire in `checkpoints.commit` so every
    `save_strategy="epoch"` checkpoint is durable before the next epoch (and
    thus resumable) starts. No-op when no hook is given (e.g. local/non-Modal
    runs), so run_training() stays usable outside Modal.
    """

    def __init__(self, hook: Callable[[], None] | None) -> None:
        self._hook = hook

    def on_save(self, args, state, control, **kwargs) -> None:  # noqa: D102
        if self._hook is not None:
            self._hook()


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
    checkpoint_commit_hook: Callable[[], None] | None = None,
) -> dict[str, Any]:
    device = pick_device()
    if device == "mps":
        print("warning: MPS selected — prefer Modal CUDA (twotower_qwen_voyage_gemini/modal_train.py)")

    output_dir = Path(cfg.output_dir)
    explicit_resume = resume_from_checkpoint
    resume_from_checkpoint = resolve_resume_checkpoint(output_dir, explicit_resume)
    if resume_from_checkpoint is not None:
        source = "explicit --resume-from-checkpoint" if resume_from_checkpoint == explicit_resume else "auto-detected from a non-empty output_dir (likely a prior preemption)"
        print(f"resuming from checkpoint ({source}): {resume_from_checkpoint}")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows, extraction_summary = load_multineg_rows(rows_path)
    train_rows, dev_rows = carve_dev(
        all_rows, seeker_fraction=dev_seeker_fraction, min_rows=dev_min_rows, seed=cfg.seed
    )
    print(f"rows: train={len(train_rows)} dev={len(dev_rows)} (source: {extraction_summary})")

    # Qwen3-Embedding-8B must load in bf16 (16GB vs 32GB fp32) and run with
    # gradient checkpointing, or micro-batch 6 does not fit. twotower.train's
    # build_model() sets neither, hence the local model.py copy.
    torch_dtype = _TORCH_DTYPES.get(cfg.extra.get("torch_dtype"))
    gradient_checkpointing = bool(cfg.extra.get("gradient_checkpointing_override"))
    model = build_model_with_dtype(
        cfg,
        device,
        torch_dtype=torch_dtype,
        gradient_checkpointing=gradient_checkpointing,
    )
    print(
        f"model: dtype={cfg.extra.get('torch_dtype')} "
        f"gradient_checkpointing={gradient_checkpointing}"
    )
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

    loss = MultipleNegativesRankingLoss(model=model)
    evaluator = MultiNegTripletDevEvaluator(dev_rows, batch_size=cfg.eval_batch_size, name="train_dev")

    padded_rows = sum(1 for r in train_rows + dev_rows if r.padded_count > 0)
    total_rows = len(train_rows) + len(dev_rows)
    meta = collect_env_metadata(cfg, device)
    meta.update(
        {
            "experiment": "twotower_qwen_voyage_gemini",
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
        callbacks=[_CommitOnSaveCallback(checkpoint_commit_hook)],
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

    # MultiNegTripletDevEvaluator writes "train_dev_beat_all_accuracy", not
    # TrainConfig's default primary_metric ("pair_auc") — select_best_checkpoint
    # keys off cfg.primary_metric, so this override is required or it silently
    # finds no matching metric and falls back to the final epoch (the same
    # failure mode documented in docs/possible-bugs.md #2).
    dev_cfg = dataclasses.replace(cfg, primary_metric="beat_all_accuracy")
    model, best_info = select_best_checkpoint_with_dtype(
        model,
        checkpoints_dir=output_dir / "checkpoints",
        cfg=dev_cfg,
        device=device,
        torch_dtype=torch_dtype,
        gradient_checkpointing=gradient_checkpointing,
    )
    print(f"selected checkpoint: {best_info}")

    final_dir = output_dir / "adapter"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))

    dev_evaluator_final = MultiNegTripletDevEvaluator(dev_rows, batch_size=cfg.eval_batch_size, name="final_dev")
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
    p.add_argument("--rows-path", type=Path, required=True)
    # defaults = twotower_qwen_bigbatch's winning micro-6 recipe
    p.add_argument("--negatives-per-anchor", type=int, default=1)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-batch-size", type=int, default=6)
    p.add_argument("--eval-batch-size", type=int, default=6)
    p.add_argument("--gradient-accumulation-steps", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--run-holdout", action="store_true", default=True)
    p.add_argument("--no-run-holdout", dest="run_holdout", action="store_false")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume-from-checkpoint", type=str, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = build_config(
        "qwen3-8b",
        run_id=args.run_id,
        rows_path=args.rows_path,
        negatives_per_anchor=args.negatives_per_anchor,
        epochs=args.epochs,
        seed=args.seed,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
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
