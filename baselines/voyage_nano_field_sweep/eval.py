"""CLI: run the full 98-combo field/query sweep locally and write results.

    python -m baselines.voyage_nano_field_sweep.eval --data-dir data

Prefer Modal for a real run (see modal_eval.py) — local MPS encoding of the
candidate corpus is slow enough per combo that 98 combos would take hours;
this entrypoint exists for smoke-testing a handful of combos via
--limit-combos, not a full sweep.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from baselines.voyage_nano_field_sweep.fields import all_combos
from baselines.voyage_nano_field_sweep.sweep import run_sweep, summarize


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--model", type=str, default="voyageai/voyage-4-nano")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--truncate-dim", type=int, default=1024)
    p.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/voyage_nano_field_sweep"))
    p.add_argument("--split-path", type=Path, default=None)
    p.add_argument(
        "--limit-combos",
        type=int,
        default=None,
        help="Only run the first N combos (smoke test, not a full sweep).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    combos = all_combos()
    if args.limit_combos:
        combos = combos[: args.limit_combos]

    results = run_sweep(
        data_dir=args.data_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        truncate_dim=args.truncate_dim,
        artifacts_dir=args.artifacts_dir,
        split_path=args.split_path,
        combos=combos,
    )

    out_path = args.artifacts_dir / "sweep_results.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {out_path} ({len(results)} combos)")
    summarize(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
