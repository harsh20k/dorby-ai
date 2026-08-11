"""Holdout (69-pair) sanity-check eval for the voyage_gemini_kl adapter.

Deliberately narrow in scope: this is a quick, cheap check that the pulled
checkpoint produces a sane, non-crashing, non-degenerate number on the real
holdout split — NOT the number this project's standing rule says decides
anything ("score on all 200 real pairs before calling anything a result, the
69-pair holdout has repeatedly overstated results in this project" — see
`docs/twotower-voyage-gemini-ctrl-experiment.md` and
`docs/twotower-kl-reg-experiment.md`, both of which saw the holdout mislead in
the optimistic direction). All-200 scoring is out of scope for this package by
design (it requires registering this run in `eval_real_full/guard.py` and
`eval_real_full/modal_eval.py`, shared files a parallel experiment is also
touching — handled separately once both land).

Reuses `twotower.eval.load_model_for_eval` and `twotower.eval.evaluate_pairs`
directly, read-only, and `twotower.data.build_split_bundle` /
`assert_no_holdout_leak` for the frozen 69-pair split — exactly the same
functions `voyage_gemini_ctrl/train.py`'s inline holdout check and
`twotower/eval.py`'s standalone CLI both use, so this number is computed
identically to every other holdout number in the project.

Usage:
  python -m twotower_voyage_gemini_kl.eval --run-id voyage_gemini_kl_001 \\
      --adapter-dir artifacts/twotower_voyage_gemini_kl/voyage_gemini_kl_001/adapter
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from baselines.metrics import print_metrics
from baselines.voyage_nano.encode import pick_device
from twotower.data import assert_no_holdout_leak, build_split_bundle
from twotower.eval import evaluate_pairs, load_model_for_eval


def run_holdout_eval(
    *,
    adapter_dir: Path,
    data_dir: Path = Path("data"),
    split_path: Path = Path("data/synthetic/seed_split.json"),
    model_name: str = "voyageai/voyage-4-nano",
    batch_size: int = 6,
    max_length: int = 4096,
    truncate_dim: int = 1024,
) -> dict:
    device = pick_device()
    bundle = build_split_bundle(data_dir, split_path)
    assert_no_holdout_leak(bundle, split_path=split_path)

    print(
        f"holdout: n={len(bundle.holdout)} "
        f"(pos={sum(1 for p in bundle.holdout if p.label == 'pos')}, "
        f"neg={sum(1 for p in bundle.holdout if p.label == 'neg')})"
    )

    model = load_model_for_eval(
        model_name=model_name,
        adapter_dir=adapter_dir,
        device=device,
        max_seq_length=max_length,
        truncate_dim=truncate_dim,
    )
    metrics = evaluate_pairs(
        model,
        bundle.holdout,
        batch_size=batch_size,
        model_name=model_name,
        device=device,
        max_length=max_length,
        truncate_dim=truncate_dim,
    )
    metrics["split"] = "holdout"
    metrics["split_hash"] = bundle.split_hash
    metrics["data_hash"] = bundle.data_hash
    metrics["adapter_dir"] = str(adapter_dir)
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--adapter-dir", type=Path, default=None, help="defaults to artifacts/twotower_voyage_gemini_kl/<run-id>/adapter")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--split-path", type=Path, default=Path("data/synthetic/seed_split.json"))
    p.add_argument("--model", type=str, default="voyageai/voyage-4-nano")
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--truncate-dim", type=int, default=1024)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    adapter_dir = args.adapter_dir or Path("artifacts/twotower_voyage_gemini_kl") / args.run_id / "adapter"
    metrics = run_holdout_eval(
        adapter_dir=adapter_dir,
        data_dir=args.data_dir,
        split_path=args.split_path,
        model_name=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        truncate_dim=args.truncate_dim,
    )
    print_metrics(metrics)

    out_dir = Path("artifacts/twotower_voyage_gemini_kl") / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "metrics_holdout_sanity.json"
    out.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
