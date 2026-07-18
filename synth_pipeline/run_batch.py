"""CLI: create seed split and run a synthetic batch through LangGraph."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

from synth_pipeline.config import FAILURE_MODES, PipelineConfig
from synth_pipeline.graph import run_one
from synth_pipeline.nodes.seed import sample_initial_state
from synth_pipeline.nodes.writer import init_manifest
from synth_pipeline.split import ensure_split
from synth_pipeline.state import Label


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate synthetic intro pairs (LangGraph + LangSmith)."
    )
    p.add_argument(
        "--init-split",
        action="store_true",
        help="Create/refresh user-disjoint seed_split.json and exit",
    )
    p.add_argument("--force-split", action="store_true", help="Overwrite existing split")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--batch-id", default=None, help="Defaults to utc timestamp")
    p.add_argument("--n-pos", type=int, default=0)
    p.add_argument("--n-neg", type=int, default=0)
    p.add_argument(
        "--failure-modes",
        default=",".join(FAILURE_MODES),
        help="Comma list; cycled for negatives",
    )
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--artifacts-dir", type=Path, default=None)
    p.add_argument("--generate-model", default=None)
    p.add_argument("--judge-model", default=None)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="No LLM calls; stub pairs for plumbing tests",
    )
    p.add_argument("--rng-seed", type=int, default=0)
    return p.parse_args(argv)


def _cfg_from_args(args: argparse.Namespace) -> PipelineConfig:
    cfg = PipelineConfig(dry_run=args.dry_run)
    if args.data_dir:
        cfg.data_dir = args.data_dir
        cfg.split_path = args.data_dir / "synthetic" / "seed_split.json"
    if args.artifacts_dir:
        cfg.artifacts_dir = args.artifacts_dir
    if args.generate_model:
        cfg.generate_model = args.generate_model
    if args.judge_model:
        cfg.judge_model = args.judge_model
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = _cfg_from_args(args)

    if args.init_split or not cfg.split_path.exists():
        payload = ensure_split(cfg, seed=args.split_seed, force=args.force_split)
        print(
            f"Wrote {cfg.split_path} split_hash={payload['split_hash']} "
            f"train_pairs={payload['n_train_pairs']} eval_pairs={payload['n_eval_pairs']}"
        )
        if args.init_split and args.n_pos == 0 and args.n_neg == 0:
            return 0

    if args.n_pos == 0 and args.n_neg == 0:
        print("Nothing to generate. Pass --n-pos / --n-neg (or --init-split only).", file=sys.stderr)
        return 2

    split = ensure_split(cfg, seed=args.split_seed, force=False)
    batch_id = args.batch_id or datetime.now(timezone.utc).strftime("batch_%Y%m%dT%H%M%SZ")
    init_manifest(cfg, batch_id, split_hash=split["split_hash"])

    modes = [m.strip() for m in args.failure_modes.split(",") if m.strip()]
    if not modes:
        modes = list(FAILURE_MODES)

    rng = random.Random(args.rng_seed)
    jobs: list[tuple[Label, str | None]] = []
    jobs.extend([("pos", None)] * args.n_pos)
    for i in range(args.n_neg):
        jobs.append(("neg", modes[i % len(modes)]))
    rng.shuffle(jobs)

    print(
        f"batch_id={batch_id} split_hash={split['split_hash']} "
        f"jobs={len(jobs)} dry_run={cfg.dry_run} "
        f"generate={cfg.generate_model} judge={cfg.judge_model}"
    )

    summary = {"staged": 0, "dropped": 0, "errors": 0}
    for i, (label, mode) in enumerate(jobs, start=1):
        try:
            initial = sample_initial_state(
                cfg=cfg,
                batch_id=batch_id,
                label=label,
                failure_mode=mode,
                rng=rng,
            )
            final = run_one(cfg, initial)
            status = final.get("status")
            if status == "staged":
                summary["staged"] += 1
            else:
                summary["dropped"] += 1
            print(
                f"[{i}/{len(jobs)}] {label} mode={mode} → {status}"
                + (f" ({final.get('drop_reason')})" if status != "staged" else "")
            )
        except Exception as exc:  # noqa: BLE001 — batch continues
            summary["errors"] += 1
            print(f"[{i}/{len(jobs)}] ERROR {label} mode={mode}: {exc}", file=sys.stderr)

    print(json.dumps({"batch_id": batch_id, **summary}, indent=2))
    print(f"artifacts: {cfg.artifacts_dir / batch_id}")
    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
