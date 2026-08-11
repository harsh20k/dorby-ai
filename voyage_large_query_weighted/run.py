"""Run the full query-weighting sweep on voyage-4-large, all 200 real pairs.

Reuses ``query_weighted.eval.run_all_arms`` unmodified via the role-adapter in
``encoder.py``. Local CLI, not Modal — voyage-4-large is an API model, no GPU
needed, and the run is small (~11.6k new tokens; everything else should be a
cache hit against ``artifacts/voyage_large_no_query``, see package docstring).

    export VOYAGE_API_KEY=pa-...
    python -m voyage_large_query_weighted.run
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
        default=Path("artifacts/voyage_large_no_query"),
        help="Shared read/write cache — profile-only and candidate texts are byte-identical "
        "to that baseline's, so this reuses its cached embeddings for free.",
    )
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/voyage_large_query_weighted/run_001"))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--subsets", default="all,train,holdout")
    args = p.parse_args(argv)

    raw_encoder = VoyageLargeEncoder(
        model_name="voyage-4-large", output_dimension=1024, cache_dir=args.cache_dir
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
