"""Lambda sensitivity sweep for the ask_offer_001 jointly-trained two-tower model.

Deliberate duplicate of baselines/reciprocal_lambda_grid_top1ctrl/eval.py's
sweep mechanics (experiment isolation rule), with one substitution: s_fwd and
s_recip come from **two independent LoRA towers** (`twotower_ask_offer`'s
`ask_offer_001` run — Ask tower on `lookingFor`(+`searchQuery`), Offer tower
on every profile field except `lookingFor`) instead of one shared frozen or
fine-tuned encoder. `twotower_ask_offer.eval.py::run_eval` already scores this pair at a
single fixed lambda (1.75, the value it was trained with); this module reuses
its exact encode/text plumbing read-only and adds the missing piece: sweeping
lambda over a grid post-hoc, the same no-fitting diagnostic every other
reciprocal_lambda_grid* package runs, so ask_offer_001 can be plotted
alongside the frozen/top1_ctrl/voyage_gemini_ctrl curves on one chart.

No fitting: lambda is swept over a fixed grid (default -2 to 2, step 0.05),
pair ROC-AUC of ``s_fwd + lambda*s_recip`` reported directly at every grid
point against the real 200 (and the 69-pair holdout). This is on top of a
model that itself trained with a *fixed* lambda (1.75) baked into its loss —
so this sweep asks "how sensitive is the trained model's score to lambda
post-hoc", not "what lambda was it trained for".

Usage:
  python -m reciprocal_lambda_grid_ask_offer.eval --adapter-dir artifacts/twotower_ask_offer/ask_offer_001/adapter
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from baselines.reciprocal_static.eval import build_lambda_grid
from baselines.reciprocal_static.text import bg_text, look_text, seeker_look_text
from baselines.voyage_nano.encode import pick_device
from eval_real_full.data import load_real_pairs
from twotower.data import LabeledPair
from twotower.train import build_model
from twotower_ask_offer.config import TOWER_CONFIG
from twotower_ask_offer.model import encode_batched

DEFAULT_LAMBDA_MIN = -2.0
DEFAULT_LAMBDA_MAX = 2.0
DEFAULT_LAMBDA_STEP = 0.05


def sweep_lambda(
    s_fwd: np.ndarray, s_recip: np.ndarray, labels: np.ndarray, lambda_grid: np.ndarray
) -> list[dict[str, float]]:
    """Pair ROC-AUC of s_fwd + lambda*s_recip at every grid point. No selection, no fitting."""
    curve: list[dict[str, float]] = []
    for lam in lambda_grid:
        combined = s_fwd + lam * s_recip
        auc = float(roc_auc_score(labels, combined))
        curve.append({"lambda": float(lam), "pair_auc": auc})
    return curve


def run_eval(
    data_dir: Path,
    split_path: Path,
    adapter_dir: Path,
    batch_size: int,
    lambda_grid: np.ndarray,
) -> dict[str, Any]:
    device = pick_device()
    print(f"device: {device}")
    print(f"adapter: {adapter_dir}")

    ask_model = build_model(TOWER_CONFIG, device)
    ask_model.load_adapter(str(adapter_dir / "ask"))
    offer_model = build_model(TOWER_CONFIG, device)
    offer_model.load_adapter(str(adapter_dir / "offer"))
    ask_model.eval()
    offer_model.eval()

    real_all = load_real_pairs(data_dir, split_path, subset="all")
    pairs: list[LabeledPair] = real_all.pairs
    print(
        f"real pairs: {len(pairs)} ({real_all.n_pos} pos / {real_all.n_neg} neg), "
        f"{real_all.n_candidates} unique candidates"
    )

    seeker_ask_texts = [seeker_look_text(lp.pair["userContactFile"], lp.pair["searchQuery"]) for lp in pairs]
    seeker_offer_texts = [bg_text(lp.pair["userContactFile"]) for lp in pairs]
    cand_ask_texts = [look_text(lp.pair["matchContactFile"]) for lp in pairs]
    cand_offer_texts = [bg_text(lp.pair["matchContactFile"]) for lp in pairs]

    torch_device = torch.device(device)
    k_seek = encode_batched(ask_model, seeker_ask_texts, role="query", cfg=TOWER_CONFIG, device=torch_device, batch_size=batch_size)
    v_seek = encode_batched(offer_model, seeker_offer_texts, role="document", cfg=TOWER_CONFIG, device=torch_device, batch_size=batch_size)
    k_cand = encode_batched(ask_model, cand_ask_texts, role="query", cfg=TOWER_CONFIG, device=torch_device, batch_size=batch_size)
    v_cand = encode_batched(offer_model, cand_offer_texts, role="document", cfg=TOWER_CONFIG, device=torch_device, batch_size=batch_size)

    s_fwd = np.einsum("ij,ij->i", k_seek, v_cand)
    s_recip = np.einsum("ij,ij->i", k_cand, v_seek)
    labels = np.array([lp.y for lp in pairs], dtype=np.int32)

    holdout_mask = np.array([lp.source == "real_holdout" for lp in pairs])
    all_mask = np.ones(len(pairs), dtype=bool)

    curves: dict[str, Any] = {}
    for name, mask in (("holdout", holdout_mask), ("all", all_mask)):
        curve = sweep_lambda(s_fwd[mask], s_recip[mask], labels[mask], lambda_grid)
        forward_only_auc = float(roc_auc_score(labels[mask], s_fwd[mask]))
        best = max(curve, key=lambda pt: pt["pair_auc"])
        curves[name] = {
            "n_pairs": int(mask.sum()),
            "forward_only_auc": forward_only_auc,
            "curve": curve,
            "best_lambda": best["lambda"],
            "best_auc": best["pair_auc"],
        }

    return {
        "model_name": "voyageai/voyage-4-nano + ask_offer_001 two LoRA towers (ask + offer)",
        "adapter_dir": str(adapter_dir),
        "device": str(device),
        "batch_size": batch_size,
        "ask_fields": ["lookingFor", "searchQuery (seeker only)"],
        "offer_fields": ["all profile fields except lookingFor (baselines.reciprocal_static.text.BG_FIELDS)"],
        "note": (
            "No fitting step: every lambda in the grid is evaluated directly "
            "against these same labels. ask_offer_001 itself was trained with "
            "a fixed lambda=1.75 baked into its loss; this sweep is a post-hoc "
            "diagnostic on top of that, not a fit or a value chosen for "
            "deployment."
        ),
        "subsets": curves,
    }


def print_summary(metrics: dict[str, Any]) -> None:
    for name in ("holdout", "all"):
        s = metrics["subsets"][name]
        print(f"\n--- {name} ({s['n_pairs']} pairs) ---")
        print(f"forward-only (lambda=0) AUC: {s['forward_only_auc']:.4f}")
        print(f"curve max: lambda={s['best_lambda']:.2f} AUC={s['best_auc']:.4f}")
        step_display = max(1, len(s["curve"]) // 20)
        for pt in s["curve"][::step_display]:
            bar = "#" * int(pt["pair_auc"] * 60)
            print(f"  lambda={pt['lambda']:+.2f}  AUC={pt['pair_auc']:.4f}  {bar}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lambda sensitivity sweep on ask_offer_001 (no fitting)")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--split-path", type=Path, default=Path("data/synthetic/seed_split.json"))
    p.add_argument("--adapter-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/reciprocal_lambda_grid_ask_offer"))
    p.add_argument("--lambda-min", type=float, default=DEFAULT_LAMBDA_MIN)
    p.add_argument("--lambda-max", type=float, default=DEFAULT_LAMBDA_MAX)
    p.add_argument("--lambda-step", type=float, default=DEFAULT_LAMBDA_STEP)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    lambda_grid = build_lambda_grid(args.lambda_min, args.lambda_max, args.lambda_step)

    metrics = run_eval(
        data_dir=args.data_dir,
        split_path=args.split_path,
        adapter_dir=args.adapter_dir,
        batch_size=args.batch_size,
        lambda_grid=lambda_grid,
    )
    print_summary(metrics)

    out_path = args.artifacts_dir / "metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
