"""Real 69-pair holdout sanity check for this package's Qwen adapter.

Self-contained: reuses twotower.eval's model loading (`run_eval_cli`,
`load_model_for_eval`) and `baselines.metrics` directly, unchanged, so the
metric shape matches every other run in this project exactly (same
`pair`/`retrieval`/`slices` keys).

This is a **holdout-only** sanity check, per the parent task split — it is
NOT the all-200-real-pairs score that this project's standing rule requires
before calling any result a win (see CLAUDE.md: "the 69-pair holdout has
repeatedly overstated results in this project", most sharply for Qwen — see
twotower_qwen_bigbatch's holdout-minus-train gap table). All-200 scoring goes
through `eval_real_full/`, registered separately.

  python -m twotower_qwen_voyage_gemini.eval \
      --adapter-dir artifacts/twotower_qwen_voyage_gemini/qwen_voyage_gemini_001/adapter
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from baselines.metrics import print_metrics
from twotower.config import TrainConfig
from twotower.eval import run_eval_cli

from twotower_qwen_voyage_gemini.config import MODEL_PRESETS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    preset = MODEL_PRESETS["qwen3-8b"]
    cfg = TrainConfig()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=cfg.data_dir)
    p.add_argument("--split-path", type=Path, default=cfg.split_path)
    p.add_argument("--adapter-dir", type=Path, required=True)
    p.add_argument("--model", type=str, default=preset["model_name"])
    p.add_argument("--batch-size", type=int, default=preset["eval_batch_size"])
    p.add_argument("--max-length", type=int, default=preset["max_seq_length"])
    p.add_argument("--truncate-dim", type=int, default=preset["truncate_dim"])
    p.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts/twotower_qwen_voyage_gemini/eval"),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    metrics = run_eval_cli(
        data_dir=args.data_dir,
        split_path=args.split_path,
        split_name="holdout",
        adapter_dir=args.adapter_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        truncate_dim=args.truncate_dim,
        artifacts_dir=args.artifacts_dir,
    )
    print_metrics(metrics)
    out = args.artifacts_dir / "metrics_holdout.json"
    out.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
