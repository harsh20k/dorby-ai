"""Same sweep as ``run.py``, but at voyage-4-large's native 2048-dim output
(no Matryoshka truncation) instead of the published 1024-dim config.

New sibling entrypoint rather than an edit to ``run.py`` — ``run.py`` produced
the published table in ``docs/twotower-no-query-experiment.md`` and stays
byte-identical. Everything else (encoder class, role adapter, ``run_all_arms``)
is reused unmodified.

    export VOYAGE_API_KEY=pa-...
    python -m voyage_large_query_weighted.run_native
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from baselines.voyage_large.encode import VoyageLargeEncoder
from query_weighted.eval import run_all_arms, write_metrics

from voyage_large_query_weighted.encoder import VoyageLargeRoleAdapter


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--split-path", type=Path, default=Path("data/synthetic/seed_split.json"))
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("artifacts/voyage_large_query_weighted_native"),
        help="Own cache dir — output_dimension=2048 is a different cache key than the "
        "published 1024-dim run, so nothing here is a cache hit.",
    )
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/voyage_large_query_weighted_native/run_001"))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--subsets", default="all,train,holdout")
    args = p.parse_args(argv)

    raw_encoder = VoyageLargeEncoder(
        model_name="voyage-4-large", output_dimension=2048, cache_dir=args.cache_dir
    )
    encoder = VoyageLargeRoleAdapter(raw_encoder)

    metrics = run_all_arms(
        encoder,
        args.data_dir,
        args.split_path,
        subsets=tuple(s.strip() for s in args.subsets.split(",") if s.strip()),
        batch_size=args.batch_size,
    )
    metrics["usage"] = {
        "total_tokens_used": raw_encoder.total_tokens_used,
        "total_api_calls": raw_encoder.total_api_calls,
        "cache_hits": raw_encoder.cache_hits,
        "cache_misses": raw_encoder.cache_misses,
    }
    write_metrics(metrics, args.out_dir)
    print(f"\nusage: {json.dumps(metrics['usage'], indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
