"""Checkpoint selection with dtype-aware reload.

twotower.train.select_best_checkpoint reloads a candidate checkpoint via its
module-level build_model(cfg, device) (always fp32) — fine for voyage-4-nano,
but would re-OOM an 8B model at the very end of a multi-hour run, after
training already completed. This is a copy of that function's logic (not an
import — the reload call site needs to be swapped, and the upstream function
gives no injection point) with the only change being the reload call, so
Arm A/B/C's twotower.train.select_best_checkpoint stays untouched.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
from sentence_transformers import SentenceTransformer

from twotower.config import TrainConfig
from twotower.train import _loud_warning
from twotower_qwen_voyage_gemini.model import build_model_with_dtype

_STEPS_RE = re.compile(r"_steps(\d+)")


def select_best_checkpoint_with_dtype(
    model: SentenceTransformer,
    *,
    checkpoints_dir: Path,
    cfg: TrainConfig,
    device: str,
    torch_dtype: torch.dtype | None = None,
    gradient_checkpointing: bool = False,
) -> tuple[SentenceTransformer, dict[str, Any]]:
    eval_dir = checkpoints_dir / "eval"
    metric_files = sorted(eval_dir.glob("train_dev_metrics_*.json"))
    if not metric_files:
        _loud_warning(
            f"select_best_checkpoint found no metric files under {eval_dir} — "
            "falling back to the FINAL epoch, not the best one."
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
        reloaded = build_model_with_dtype(
            cfg, device, torch_dtype=torch_dtype, gradient_checkpointing=gradient_checkpointing
        )
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
