"""LoRA fine-tune voyage-4-nano with pairwise OnlineContrastiveLoss."""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
try:
    from sentence_transformers.sentence_transformer.losses import ContrastiveLoss
    from sentence_transformers.sentence_transformer.training_args import BatchSamplers
except ImportError:  # ST 3.x
    from sentence_transformers.losses import ContrastiveLoss
    from sentence_transformers.training_args import BatchSamplers

from baselines.voyage_nano.encode import pick_device
from twotower.config import TrainConfig, resolve_mixed_precision
from twotower.data import (
    assert_no_holdout_leak,
    build_split_bundle,
    pairs_to_hf_dict,
)
from twotower.eval import PairMetricsEvaluator, evaluate_pairs


def discover_linear_suffixes(model: SentenceTransformer) -> dict[str, int]:
    """Count nn.Linear modules by suffix under the backbone auto_model."""
    auto = model[0].auto_model
    counts: dict[str, int] = {}
    for name, module in auto.named_modules():
        if isinstance(module, torch.nn.Linear):
            suffix = name.rsplit(".", 1)[-1]
            counts[suffix] = counts.get(suffix, 0) + 1
    return counts


def validate_lora_targets(model: SentenceTransformer, cfg: TrainConfig) -> dict[str, int]:
    counts = discover_linear_suffixes(model)
    missing = [t for t in cfg.lora_target_modules if counts.get(t, 0) == 0]
    if missing:
        raise RuntimeError(f"LoRA targets not found in backbone: {missing}; saw {counts}")
    for t in cfg.lora_target_modules:
        n = counts[t]
        if n != cfg.expected_layers_per_target:
            raise RuntimeError(
                f"expected {cfg.expected_layers_per_target} modules for {t}, found {n}"
            )
    return {t: counts[t] for t in cfg.lora_target_modules}


def add_lora_adapter(model: SentenceTransformer, cfg: TrainConfig) -> dict[str, Any]:
    target_counts = validate_lora_targets(model, cfg)
    lora = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        target_modules=list(cfg.lora_target_modules),
    )
    model.add_adapter(lora)

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable parameters after add_adapter")
    bad = [n for n in trainable if "lora_" not in n.lower()]
    if bad:
        raise RuntimeError(f"non-LoRA trainable params: {bad[:10]}")

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    return {
        "target_counts": target_counts,
        "n_trainable": n_trainable,
        "n_total": n_total,
        "trainable_pct": round(100.0 * n_trainable / max(n_total, 1), 4),
        "trainable_name_sample": trainable[:8],
    }


def truncation_stats(
    model: SentenceTransformer,
    texts: list[str],
    max_seq_length: int,
) -> dict[str, Any]:
    tok = model.tokenizer
    lengths = [len(tok.encode(t, add_special_tokens=True)) for t in texts]
    arr = np.asarray(lengths, dtype=np.int32)
    return {
        "n": int(arr.size),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": int(arr.max()) if arr.size else 0,
        "frac_over_max_seq": float(np.mean(arr > max_seq_length)) if arr.size else 0.0,
        "max_seq_length": max_seq_length,
    }


def _features_to_device(features: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in features.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def smoke_backward(model: SentenceTransformer, cfg: TrainConfig) -> None:
    """One tiny forward/backward to confirm LoRA grads are non-zero."""
    del cfg  # unused; kept for call-site symmetry
    model.train()
    device = next(model.parameters()).device
    # ST >=5 prefers preprocess(); tokenize() may return mixed non-tensor fields.
    preprocess = getattr(model, "preprocess", None) or model.tokenize
    feats_q = _features_to_device(dict(preprocess(["smoke query about fundraising"])), device)
    feats_d = _features_to_device(
        dict(preprocess(["smoke document about venture partners"])), device
    )
    emb_q = model(feats_q)["sentence_embedding"]
    emb_d = model(feats_d)["sentence_embedding"]
    loss = 1.0 - torch.nn.functional.cosine_similarity(emb_q, emb_d).mean()
    model.zero_grad(set_to_none=True)
    loss.backward()
    for name, p in model.named_parameters():
        if (
            p.requires_grad
            and p.grad is not None
            and "lora_" in name.lower()
            and float(p.grad.detach().abs().sum()) > 0
        ):
            return
    raise RuntimeError("smoke_backward: no non-zero LoRA gradients")


_STEPS_RE = re.compile(r"_steps(\d+)")


def _loud_warning(msg: str) -> None:
    banner = "!" * 78
    print(f"\n{banner}\nWARNING: {msg}\n{banner}\n")


def select_best_checkpoint(
    model: SentenceTransformer,
    *,
    checkpoints_dir: Path,
    cfg: TrainConfig,
    device: str,
) -> tuple[SentenceTransformer, dict[str, Any]]:
    """Pick best epoch checkpoint by train-dev primary metric; reload if needed.

    ST's Trainer writes evaluator output under ``checkpoints_dir/eval/``, not
    ``checkpoints_dir`` itself (see sentence_transformers/base/trainer.py's
    ``evaluate()`` — output_path = os.path.join(args.output_dir, "eval")).
    Selection is keyed on ``steps`` (parsed from the filename), which maps
    exactly to HF's ``checkpoint-{global_step}`` directory names — the only
    stable join key. ``epoch`` is a float from the Trainer (e.g. ``1.0``) and
    is not used for matching, only for logging.
    """
    eval_dir = checkpoints_dir / "eval"
    metric_files = sorted(eval_dir.glob("train_dev_metrics_*.json"))
    if not metric_files:
        _loud_warning(
            f"select_best_checkpoint found no metric files under {eval_dir} — "
            "falling back to the FINAL epoch, not the best one. The "
            "checkpoint-selection safety net did not run for this training "
            "run; do not assume the shipped adapter is the best epoch."
        )
        return model, {"source": "final_in_memory", "reason": "no_metric_files", "eval_dir": str(eval_dir)}

    ckpts_by_step = {
        int(p.name.split("-")[-1]): p
        for p in checkpoints_dir.iterdir()
        if p.is_dir() and p.name.startswith("checkpoint-")
    }

    best_path: Path | None = None
    best_score = float("-inf")
    best_steps: int | None = None
    key = f"train_dev_{cfg.primary_metric}"
    skipped_unparseable = 0
    for path in metric_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        flat = payload.get("flat") or {}
        if key not in flat:
            continue
        score = float(flat[key])
        m = _STEPS_RE.search(path.stem)
        if m is None:
            skipped_unparseable += 1
            continue
        steps = int(m.group(1))
        if score > best_score:
            best_score = score
            best_steps = steps
            best_path = path

    if skipped_unparseable:
        _loud_warning(
            f"select_best_checkpoint could not parse `steps` from "
            f"{skipped_unparseable} metric file(s) under {eval_dir} — those "
            "were skipped as candidates for best-checkpoint selection."
        )

    chosen = ckpts_by_step.get(best_steps) if best_steps is not None else None
    if chosen is None:
        _loud_warning(
            f"select_best_checkpoint found a best score ({best_score}) at "
            f"steps={best_steps} but no matching checkpoint-{best_steps} "
            f"directory under {checkpoints_dir} (available: "
            f"{sorted(ckpts_by_step)}) — falling back to the FINAL epoch."
        )
        return model, {
            "source": "final_in_memory",
            "reason": "checkpoint_dir_not_found",
            "best_score": best_score,
            "best_steps": best_steps,
            "metric_file": str(best_path) if best_path else None,
            "available_checkpoint_steps": sorted(ckpts_by_step),
        }

    try:
        # Intermediate Trainer checkpoints are LoRA-only saves (adapter_model
        # .safetensors + adapter_config.json, no base-model config.json /
        # preprocessor_config.json) — same shape as the final adapter/ dir.
        # Rebuild the base model fresh and load the adapter onto it, mirroring
        # eval.py::load_model_for_eval, rather than reconstructing a full
        # SentenceTransformer directly from the checkpoint path (which fails:
        # "Can't load feature extractor" / missing preprocessor_config.json).
        reloaded = build_model(cfg, device)
        reloaded.load_adapter(str(chosen))
        return reloaded, {
            "source": "checkpoint",
            "path": str(chosen),
            "best_score": best_score,
            "best_steps": best_steps,
            "metric_file": str(best_path) if best_path else None,
        }
    except Exception as exc:  # noqa: BLE001
        _loud_warning(f"failed to reload best checkpoint {chosen}: {exc}")
        return model, {
            "source": "final_in_memory",
            "reason": "reload_failed",
            "error": str(exc),
            "attempted_path": str(chosen),
            "best_score": best_score,
            "best_steps": best_steps,
        }


def collect_env_metadata(cfg: TrainConfig, device: str) -> dict[str, Any]:
    import datasets
    import peft
    import sentence_transformers
    import transformers

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": device,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "transformers": transformers.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "peft": peft.__version__,
        "datasets": datasets.__version__,
        "config": cfg.to_dict(),
    }


def build_model(cfg: TrainConfig, device: str) -> SentenceTransformer:
    model = SentenceTransformer(
        cfg.model_name,
        trust_remote_code=cfg.trust_remote_code,
        truncate_dim=cfg.truncate_dim,
        device=device,
        revision=cfg.model_revision,
        prompts={
            "query": cfg.query_prompt,
            "document": cfg.document_prompt,
        },
    )
    model.max_seq_length = cfg.max_seq_length
    if cfg.gradient_checkpointing:
        try:
            auto = model[0].auto_model
            if hasattr(auto, "gradient_checkpointing_enable"):
                auto.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            if hasattr(auto, "config"):
                auto.config.use_cache = False
        except Exception as exc:  # noqa: BLE001 — best-effort on custom remote code
            print(f"warning: gradient checkpointing not enabled: {exc}")
    return model


def run_training(
    cfg: TrainConfig,
    *,
    dry_run: bool = False,
    resume_from_checkpoint: str | None = None,
) -> dict[str, Any]:
    device = pick_device()
    # Prefer CUDA for training; warn on MPS.
    if device == "mps":
        print(
            "warning: MPS selected — LoRA backward may be slow/incorrect; "
            "prefer Modal CUDA (twotower/modal_train.py)"
        )

    output_dir = Path(cfg.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not resume_from_checkpoint:
        raise FileExistsError(
            f"output_dir {output_dir} is non-empty; pass a fresh run id or resume"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = build_split_bundle(
        cfg.data_dir,
        cfg.split_path,
        train_dev_user_fraction=cfg.train_dev_user_fraction,
        train_dev_min_pairs=cfg.train_dev_min_pairs,
        seed=cfg.seed,
        include_synth=cfg.include_synth,
    )
    assert_no_holdout_leak(bundle, split_path=cfg.split_path)
    print(f"data counts: {bundle.counts}")
    print(f"split_hash={bundle.split_hash} data_hash={bundle.data_hash}")

    model = build_model(cfg, device)
    lora_info = add_lora_adapter(model, cfg)
    print(f"lora: {lora_info}")

    all_texts = [p.seeker_text for p in bundle.train] + [
        p.candidate_text for p in bundle.train
    ]
    trunc = truncation_stats(model, all_texts, cfg.max_seq_length)
    print(f"truncation: {trunc}")

    smoke_backward(model, cfg)
    print("smoke_backward: ok")

    train_ds = Dataset.from_dict(pairs_to_hf_dict(bundle.train))
    # Keep column order: anchor, other, label
    train_ds = train_ds.select_columns(["anchor", "other", "label"])

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
        # HF load_best_model_at_end fails on voyage-4-nano remote code
        # ("Unrecognized model" / missing model_type). We pick the best
        # checkpoint ourselves after training via train-dev metric files.
        load_best_model_at_end=False,
        logging_steps=cfg.logging_steps,
        seed=cfg.seed,
        data_seed=cfg.seed,
        # Column -> prompt *strings* (data collator prepends them directly).
        prompts={"anchor": cfg.query_prompt, "other": cfg.document_prompt},
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        report_to=[],
        run_name=output_dir.name,
    )

    # ContrastiveLoss is robust to single-label mini-batches (unlike Online*).
    loss = ContrastiveLoss(model=model, margin=cfg.contrastive_margin)
    evaluator = PairMetricsEvaluator(
        bundle.train_dev,
        batch_size=cfg.eval_batch_size,
        name="train_dev",
        primary_metric=cfg.primary_metric,
    )

    meta = collect_env_metadata(cfg, device)
    meta.update(
        {
            "split_hash": bundle.split_hash,
            "data_hash": bundle.data_hash,
            "counts": bundle.counts,
            "lora": lora_info,
            "truncation": trunc,
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

    model, best_info = select_best_checkpoint(
        model,
        checkpoints_dir=output_dir / "checkpoints",
        cfg=cfg,
        device=device,
    )
    print(f"selected checkpoint: {best_info}")

    final_dir = output_dir / "adapter"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))

    # Train-dev metrics for the selected checkpoint.
    dev_metrics = evaluate_pairs(
        model,
        bundle.train_dev,
        batch_size=cfg.eval_batch_size,
        model_name=cfg.model_name,
        device=device,
        max_length=cfg.max_seq_length,
        truncate_dim=cfg.truncate_dim,
    )
    (output_dir / "metrics_train_dev.json").write_text(
        json.dumps(dev_metrics, indent=2) + "\n"
    )

    holdout_metrics = None
    if cfg.run_holdout:
        holdout_metrics = evaluate_pairs(
            model,
            bundle.holdout,
            batch_size=cfg.eval_batch_size,
            model_name=cfg.model_name,
            device=device,
            max_length=cfg.max_seq_length,
            truncate_dim=cfg.truncate_dim,
        )
        (output_dir / "metrics_holdout.json").write_text(
            json.dumps(holdout_metrics, indent=2) + "\n"
        )

    result = {
        **meta,
        "adapter_dir": str(final_dir),
        "best_checkpoint": best_info,
        "metrics_train_dev": {
            "pair_auc": dev_metrics["pair"]["roc_auc"],
            "pair_ap": dev_metrics["pair"]["average_precision"],
            "retrieval_mrr": dev_metrics["retrieval"]["mrr"],
        },
        "metrics_holdout": (
            {
                "pair_auc": holdout_metrics["pair"]["roc_auc"],
                "pair_ap": holdout_metrics["pair"]["average_precision"],
                "retrieval_mrr": holdout_metrics["retrieval"]["mrr"],
            }
            if holdout_metrics
            else None
        ),
    }
    (output_dir / "run_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"saved adapter -> {final_dir}")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    cfg = TrainConfig()
    p = argparse.ArgumentParser(description="LoRA two-tower fine-tune (voyage-4-nano)")
    p.add_argument("--data-dir", type=Path, default=cfg.data_dir)
    p.add_argument("--split-path", type=Path, default=cfg.split_path)
    p.add_argument("--output-dir", type=Path, default=cfg.output_dir)
    p.add_argument("--model", type=str, default=cfg.model_name)
    p.add_argument("--model-revision", type=str, default=None)
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--lora-rank", type=int, default=cfg.lora_rank)
    p.add_argument("--lora-alpha", type=int, default=cfg.lora_alpha)
    p.add_argument("--max-seq-length", type=int, default=cfg.max_seq_length)
    p.add_argument("--train-batch-size", type=int, default=cfg.train_batch_size)
    p.add_argument("--eval-batch-size", type=int, default=cfg.eval_batch_size)
    p.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=cfg.gradient_accumulation_steps,
    )
    p.add_argument("--learning-rate", type=float, default=cfg.learning_rate)
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--truncate-dim", type=int, default=cfg.truncate_dim)
    p.add_argument("--run-holdout", action="store_true")
    p.add_argument(
        "--real-only",
        action="store_true",
        help="Exclude promoted synthetic pairs from the train pool (control arm — "
        "see docs/twotower-run-001-findings.md). This is now the default; the flag "
        "is kept so `arm_a_real_only`'s original command line still reproduces.",
    )
    p.add_argument(
        "--include-synth",
        action="store_true",
        help="Opt back into the 460 quarantined batch_500_001 synthetic pairs. "
        "They are known-harmful (candidate-only classifier 99.2%% accurate on "
        "them; run_001 scored 0.4845 hard-neg AUC, below chance) — see "
        "data/archive/batch_500_001_quarantined/README.md. Needed only to "
        "reproduce run_001.",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume-from-checkpoint", type=str, default=None)
    return p.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        model_name=args.model,
        model_revision=args.model_revision,
        epochs=args.epochs,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        max_seq_length=args.max_seq_length,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
        truncate_dim=args.truncate_dim,
        data_dir=args.data_dir,
        split_path=args.split_path,
        output_dir=args.output_dir,
        run_holdout=args.run_holdout,
        # Quarantined by default: synth pairs enter only on an explicit opt-in,
        # and --real-only still wins if both are passed.
        include_synth=args.include_synth and not args.real_only,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = config_from_args(args)
    run_training(
        cfg,
        dry_run=args.dry_run,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
