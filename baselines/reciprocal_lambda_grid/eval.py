"""Lambda sensitivity sweep for the static reciprocal score — no fitting step.

Both prior reciprocal experiments (baselines/reciprocal_static,
baselines/reciprocal_static_rrf003) picked a single lambda by grid-search
*fitting* it to maximize AUC on some population, then froze it and evaluated
elsewhere. This experiment does neither: it sweeps lambda over a fixed range
(default -2 to 2, step 0.05) and reports pair ROC-AUC of
`s_fwd + lambda*s_recip` directly against the real 200 (and, for reference,
just the 69-pair holdout) at *every* grid point. There is no train/holdout
split here and no "best lambda" being selected for deployment — this is a
diagnostic curve, not a fit. Read it as "how sensitive is real-200 AUC to
lambda, and does any value clearly beat lambda=0 (forward-only)."

bg_text is positioning + background only (baselines/reciprocal_lambda_grid/
text.py), matching the field choice used in
baselines/reciprocal_static_rrf003 — not the original reciprocal_static's
wider field set.

Usage:
  python -m baselines.reciprocal_lambda_grid.eval --data-dir data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from baselines.reciprocal_lambda_grid.text import bg_text, look_text, seeker_look_text
from baselines.reciprocal_static.eval import build_lambda_grid
from baselines.voyage_nano.encode import VoyageNanoEncoder, cosine_scores, pick_device
from eval_real_full.data import load_real_pairs
from twotower.data import LabeledPair

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
    model_name: str,
    batch_size: int,
    max_length: int,
    truncate_dim: int | None,
    artifacts_dir: Path,
    lambda_grid: np.ndarray,
) -> dict[str, Any]:
    device = pick_device()
    print(f"device: {device}")
    print(f"model:  {model_name}")

    real_all = load_real_pairs(data_dir, split_path, subset="all")
    pairs = real_all.pairs
    print(
        f"real pairs: {len(pairs)} ({real_all.n_pos} pos / {real_all.n_neg} neg), "
        f"{real_all.n_candidates} unique candidates"
    )

    encoder = VoyageNanoEncoder(
        model_name=model_name,
        device=device,
        max_length=max_length,
        truncate_dim=truncate_dim,
        cache_dir=artifacts_dir,
    )

    seeker_look_texts = [seeker_look_text(lp.pair["userContactFile"], lp.pair["searchQuery"]) for lp in pairs]
    seeker_bg_texts = [bg_text(lp.pair["userContactFile"]) for lp in pairs]
    cand_look_texts = [look_text(lp.pair["matchContactFile"]) for lp in pairs]
    cand_bg_texts = [bg_text(lp.pair["matchContactFile"]) for lp in pairs]

    seeker_look_emb = encoder.encode(seeker_look_texts, role="query", batch_size=batch_size, cache_name="all_seeker_look")
    seeker_bg_emb = encoder.encode(seeker_bg_texts, role="document", batch_size=batch_size, cache_name="all_seeker_bg")
    cand_look_emb = encoder.encode(cand_look_texts, role="query", batch_size=batch_size, cache_name="all_cand_look")
    cand_bg_emb = encoder.encode(cand_bg_texts, role="document", batch_size=batch_size, cache_name="all_cand_bg")

    s_fwd = cosine_scores(seeker_look_emb, cand_bg_emb)
    s_recip = cosine_scores(cand_look_emb, seeker_bg_emb)
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
        "model_name": model_name,
        "device": str(device),
        "max_length": max_length,
        "truncate_dim": truncate_dim,
        "batch_size": batch_size,
        "bg_fields": ["positioning", "background"],
        "lambda_grid": {
            "min": float(lambda_grid.min()),
            "max": float(lambda_grid.max()),
            "step": float(lambda_grid[1] - lambda_grid[0]) if len(lambda_grid) > 1 else 0.0,
            "n_points": len(lambda_grid),
        },
        "note": (
            "No fitting step: every lambda in the grid is evaluated directly "
            "against these same labels. best_lambda/best_auc describe the "
            "curve's shape (how sensitive AUC is to lambda), not a value "
            "chosen for deployment via a leakage-safe fit."
        ),
        "subsets": curves,
    }


def print_summary(metrics: dict[str, Any]) -> None:
    lg = metrics["lambda_grid"]
    print(f"\n=== lambda sweep: [{lg['min']}, {lg['max']}] step {lg['step']} ({lg['n_points']} points) ===")
    print("(no fitting — every point evaluated directly against the labels shown)")
    for name in ("holdout", "all"):
        s = metrics["subsets"][name]
        print(f"\n--- {name} ({s['n_pairs']} pairs) ---")
        print(f"forward-only (lambda=0) AUC: {s['forward_only_auc']:.4f}")
        print(f"curve max: lambda={s['best_lambda']:.2f} AUC={s['best_auc']:.4f}")
        # print a coarse view of the curve so the shape is visible without opening the JSON
        step_display = max(1, len(s["curve"]) // 20)
        for pt in s["curve"][::step_display]:
            bar = "#" * int(pt["pair_auc"] * 60)
            print(f"  lambda={pt['lambda']:+.2f}  AUC={pt['pair_auc']:.4f}  {bar}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lambda sensitivity sweep for the static reciprocal score (no fitting)")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--split-path", type=Path, default=Path("data/synthetic/seed_split.json"))
    p.add_argument("--model", type=str, default="voyageai/voyage-4-nano")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--truncate-dim", type=int, default=1024)
    p.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/reciprocal_lambda_grid"))
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
        model_name=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        truncate_dim=args.truncate_dim,
        artifacts_dir=args.artifacts_dir,
        lambda_grid=lambda_grid,
    )
    print_summary(metrics)

    out_path = args.artifacts_dir / "metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
