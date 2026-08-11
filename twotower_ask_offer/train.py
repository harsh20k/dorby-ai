"""Hand-rolled training loop for the two-tower reciprocal fine-tune.

Not SentenceTransformerTrainer-based — see model.py's docstring for why: two
independently-weighted models optimized jointly against one loss that mixes
both of their outputs doesn't fit that abstraction. This loop is the direct
equivalent for two models: shuffle -> batch -> forward both towers -> compute
S -> backward -> step, with the same per-epoch checkpoint + dev-eval +
best-checkpoint-by-recall@1 discipline as every other twotower_* package
(never silently ship the final epoch — docs/possible-bugs.md #2, hit three
times in this project before it was fixed).

Known scope simplification vs. the Trainer-based packages: no mixed precision
(fp32 throughout). LoRA on a small base model over ~2700 train rows / 5
epochs is cheap enough that this doesn't need bf16/fp16 to be Modal-feasible;
noted here rather than silently claiming parity with the other packages'
training args.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import get_linear_schedule_with_warmup

from baselines.voyage_nano.encode import pick_device
from twotower.data import assert_no_holdout_leak, build_split_bundle
from twotower.eval import evaluate_pairs
from twotower.train import collect_env_metadata, truncation_stats

from twotower_ask_offer.config import AskOfferConfig
from twotower_ask_offer.data import AskOfferRow, carve_dev, load_rows
from twotower_ask_offer.eval_dev import evaluate_dev
from twotower_ask_offer.loss import build_batch_texts, combine_and_cross_entropy
from twotower_ask_offer.model import build_two_towers, encode_texts, trainable_parameters


def _batches(rows: list[AskOfferRow], batch_size: int, seed: int) -> list[list[AskOfferRow]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    return [shuffled[i : i + batch_size] for i in range(0, len(shuffled), batch_size)]


def smoke_backward(ask_model, offer_model, cfg: AskOfferConfig, device: torch.device) -> None:
    """One tiny forward/backward through both towers + the reciprocal loss to
    confirm LoRA grads are non-zero on both, before spending real steps."""
    ask_model.train()
    offer_model.train()
    # >=2 anchors/pool items: with a single class, softmax is always 1.0 and
    # cross-entropy's gradient is exactly zero regardless of the logit — that
    # would look like a broken loss when it's really just a degenerate batch.
    seeker_ask = ["lookingFor: fundraising partners", "lookingFor: engineering hires"]
    seeker_offer = ["positioning: builds robots", "positioning: runs a bakery"]
    pool_ask = ["lookingFor: investors", "lookingFor: candidates"]
    pool_offer = ["positioning: VC fund", "positioning: staffing agency"]
    k_seek = encode_texts(ask_model, seeker_ask, role="query", cfg=cfg.tower, device=device)
    v_seek = encode_texts(offer_model, seeker_offer, role="document", cfg=cfg.tower, device=device)
    k_pool = encode_texts(ask_model, pool_ask, role="query", cfg=cfg.tower, device=device)
    v_pool = encode_texts(offer_model, pool_offer, role="document", cfg=cfg.tower, device=device)
    loss, _ = combine_and_cross_entropy(k_seek, v_seek, k_pool, v_pool, lam=cfg.lam, scale=cfg.loss_scale)
    ask_model.zero_grad(set_to_none=True)
    offer_model.zero_grad(set_to_none=True)
    loss.backward()

    def has_lora_grad(model) -> bool:
        return any(
            p.requires_grad and p.grad is not None and "lora_" in n.lower() and float(p.grad.detach().abs().sum()) > 0
            for n, p in model.named_parameters()
        )

    if not has_lora_grad(ask_model):
        raise RuntimeError("smoke_backward: no non-zero LoRA gradients on ask_model")
    if not has_lora_grad(offer_model):
        raise RuntimeError("smoke_backward: no non-zero LoRA gradients on offer_model")
    ask_model.zero_grad(set_to_none=True)
    offer_model.zero_grad(set_to_none=True)


def _save_towers(ask_model, offer_model, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ask_model.save_pretrained(str(out_dir / "ask"))
    offer_model.save_pretrained(str(out_dir / "offer"))


def _load_towers(cfg: AskOfferConfig, device: str, adapter_dir: Path):
    from twotower.train import build_model

    ask_model = build_model(cfg.tower, device)
    ask_model.load_adapter(str(adapter_dir / "ask"))
    offer_model = build_model(cfg.tower, device)
    offer_model.load_adapter(str(adapter_dir / "offer"))
    return ask_model, offer_model


def run_training(
    cfg: AskOfferConfig,
    *,
    dry_run: bool = False,
    run_holdout: bool = True,
    real_holdout_data_dir: Path = Path("data"),
    real_holdout_split_path: Path = Path("data/synthetic/seed_split.json"),
    max_rows: int | None = None,
) -> dict[str, Any]:
    device = pick_device()
    if device == "mps":
        print("warning: MPS selected — prefer Modal CUDA (twotower_ask_offer/modal_train.py)")

    run_dir = cfg.run_dir
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"run_dir {run_dir} is non-empty; pass a fresh run id")
    run_dir.mkdir(parents=True, exist_ok=True)

    rows, provenance = load_rows(cfg.rows_path)
    if max_rows is not None:
        rows = rows[:max_rows]
    train_rows, dev_rows = carve_dev(
        rows, seeker_fraction=cfg.dev_seeker_fraction, min_rows=cfg.dev_min_rows, seed=cfg.seed
    )
    print(f"rows: train={len(train_rows)} dev={len(dev_rows)} (source rows: {len(rows)})")

    ask_model, offer_model, lora_info = build_two_towers(cfg.tower, device)
    print(f"lora: {lora_info}")

    sample_texts = [seeker_look_text_sample(r) for r in train_rows[:200]]
    trunc = truncation_stats(ask_model, sample_texts, cfg.tower.max_seq_length) if sample_texts else {}
    print(f"truncation (ask, sample): {trunc}")

    smoke_backward(ask_model, offer_model, cfg, torch.device(device))
    print("smoke_backward: ok")

    optimizer = torch.optim.AdamW(
        trainable_parameters(ask_model, offer_model),
        lr=cfg.tower.learning_rate,
        weight_decay=cfg.tower.weight_decay,
    )

    batch_size = cfg.tower.train_batch_size
    grad_accum = cfg.tower.gradient_accumulation_steps
    n_micro_batches_per_epoch = max(1, (len(train_rows) + batch_size - 1) // batch_size)
    n_optimizer_steps_per_epoch = max(1, (n_micro_batches_per_epoch + grad_accum - 1) // grad_accum)
    total_optimizer_steps = n_optimizer_steps_per_epoch * cfg.epochs
    n_warmup = max(1, int(total_optimizer_steps * cfg.tower.warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=n_warmup, num_training_steps=total_optimizer_steps)

    loss_history: list[dict[str, Any]] = []
    dev_history: list[dict[str, Any]] = []
    checkpoints_dir = run_dir / "checkpoints"

    epochs = 1 if dry_run else cfg.epochs
    for epoch in range(1, epochs + 1):
        ask_model.train()
        offer_model.train()
        batches = _batches(train_rows, batch_size, seed=cfg.seed + epoch)
        if dry_run:
            batches = batches[:2]
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(batches, start=1):
            bt = build_batch_texts(batch)
            device_t = torch.device(device)
            k_seek = encode_texts(ask_model, bt.seeker_ask, role="query", cfg=cfg.tower, device=device_t)
            v_seek = encode_texts(offer_model, bt.seeker_offer, role="document", cfg=cfg.tower, device=device_t)
            k_pool = encode_texts(ask_model, bt.pool_ask, role="query", cfg=cfg.tower, device=device_t)
            v_pool = encode_texts(offer_model, bt.pool_offer, role="document", cfg=cfg.tower, device=device_t)
            loss, _ = combine_and_cross_entropy(k_seek, v_seek, k_pool, v_pool, lam=cfg.lam, scale=cfg.loss_scale)
            (loss / grad_accum).backward()

            if step % grad_accum == 0 or step == len(batches):
                torch.nn.utils.clip_grad_norm_(trainable_parameters(ask_model, offer_model), cfg.tower.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            loss_history.append({"epoch": epoch, "step": step, "loss": float(loss.detach().cpu())})
            if step % 20 == 0 or step == len(batches):
                print(f"  epoch {epoch} step {step}/{len(batches)} loss={float(loss.detach().cpu()):.4f}")

        epoch_dir = checkpoints_dir / f"epoch_{epoch}"
        ask_model.eval()
        offer_model.eval()
        _save_towers(ask_model, offer_model, epoch_dir)
        dev_metrics = evaluate_dev(
            ask_model, offer_model, dev_rows, lam=cfg.lam, cfg=cfg.tower, device=torch.device(device),
            batch_size=cfg.tower.eval_batch_size, name="train_dev", output_path=checkpoints_dir / "eval", epoch=epoch,
        )
        dev_history.append({"epoch": epoch, **dev_metrics})

    best = max(dev_history, key=lambda d: d.get("train_dev_recall@1", 0.0)) if dev_history else None
    if best is not None:
        best_epoch = best["epoch"]
        print(f"selected checkpoint: epoch {best_epoch} (train_dev_recall@1={best.get('train_dev_recall@1'):.4f})")
        ask_model, offer_model = _load_towers(cfg, device, checkpoints_dir / f"epoch_{best_epoch}")
        selection_info = {"source": "checkpoint", "epoch": best_epoch, "metric": best.get("train_dev_recall@1")}
    else:
        selection_info = {"source": "final_in_memory", "reason": "no_dev_rows"}

    final_dir = run_dir / "adapter"
    ask_model.eval()
    offer_model.eval()
    _save_towers(ask_model, offer_model, final_dir)

    meta = collect_env_metadata(cfg.tower, device)
    meta.update(
        {
            "experiment": "twotower_ask_offer",
            "run_id": cfg.run_id,
            "config": cfg.to_dict(),
            "rows_provenance": provenance,
            "n_train_rows": len(train_rows),
            "n_dev_rows": len(dev_rows),
            "lora": lora_info,
            "truncation_sample": trunc,
            "loss": "reciprocal_in_batch (combine_and_cross_entropy)",
            "selection": selection_info,
            "dry_run": dry_run,
        }
    )
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (run_dir / "loss_history.json").write_text(json.dumps({"loss_history": loss_history, "dev_history": dev_history}, indent=2) + "\n")

    holdout_metrics = None
    if run_holdout and not dry_run:
        from twotower_ask_offer.eval import evaluate_ask_offer_pairs

        bundle = build_split_bundle(real_holdout_data_dir, real_holdout_split_path)
        assert_no_holdout_leak(bundle, split_path=real_holdout_split_path)
        holdout_metrics = evaluate_ask_offer_pairs(
            ask_model, offer_model, bundle.holdout, lam=cfg.lam, cfg=cfg.tower, device=torch.device(device),
            batch_size=cfg.tower.eval_batch_size,
        )
        (run_dir / "metrics_holdout.json").write_text(json.dumps(holdout_metrics, indent=2) + "\n")

    result = {
        **meta,
        "adapter_dir": str(final_dir),
        "metrics_train_dev": best,
        "metrics_holdout": holdout_metrics,
    }
    (run_dir / "run_result.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(f"saved adapters -> {final_dir}")
    return result


def seeker_look_text_sample(row: AskOfferRow) -> str:
    from baselines.reciprocal_static.text import seeker_look_text

    return seeker_look_text(row.seeker_profile, row.search_query)


def parse_args(argv: list[str] | None = None):
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lam", type=float, default=1.75)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rows-path", type=Path, default=Path("artifacts/twotower_ask_offer/ask_offer_rows.json"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-rows", type=int, default=None, help="cap rows loaded (dry-run/smoke use)")
    p.add_argument("--run-holdout", action="store_true", default=True)
    p.add_argument("--no-run-holdout", dest="run_holdout", action="store_false")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from twotower_ask_offer.config import build_config

    args = parse_args(argv)
    cfg = build_config(
        run_id=args.run_id,
        rows_path=args.rows_path,
        lam=args.lam,
        epochs=args.epochs,
        seed=args.seed,
    )
    result = run_training(cfg, dry_run=args.dry_run, run_holdout=args.run_holdout, max_rows=args.max_rows)
    print(json.dumps({k: v for k, v in result.items() if k not in ("config",)}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
