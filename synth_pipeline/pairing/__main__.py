"""CLI: python -m synth_pipeline.pairing --profile-run <dir> --batch-id <id>"""

from __future__ import annotations

import argparse
from pathlib import Path

from synth_pipeline.pairing.bedrock import DEFAULT_MODEL_ID, DEFAULT_REGION
from synth_pipeline.pairing.run import run_pairing

REPO_ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile-run", type=Path, required=True,
                    help="a bedrock_profile_gen.py / local_gemma_profile_gen.py run dir")
    ap.add_argument("--batch-id", required=True)
    ap.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    ap.add_argument("--artifacts-dir", type=Path, default=REPO_ROOT / "artifacts")
    ap.add_argument("--split-path", type=Path, default=None)
    ap.add_argument("--queries-per-profile", type=int, default=2)
    ap.add_argument("--k-per-query", type=int, default=5)
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--region", default=DEFAULT_REGION)
    ap.add_argument("--label-mode", choices=("quantile", "absolute"), default="quantile",
                    help="quantile: split the batch by its own score distribution "
                         "(default). absolute: use the real-pair threshold — measured "
                         "NOT to transfer to synthetic batches, kept for comparison")
    ap.add_argument("--pos-frac", type=float, default=0.3,
                    help="quantile mode: top fraction labeled pos")
    ap.add_argument("--neg-frac", type=float, default=0.3,
                    help="quantile mode: bottom fraction labeled neg")
    ap.add_argument("--deadband-margin", type=float, default=0.25,
                    help="absolute mode: half-width of the unlabeled band, in "
                         "fit-score std units")
    ap.add_argument("--fusion-mode", choices=("alpha", "logistic"), default="alpha")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap profiles loaded (debugging)")
    ap.add_argument("--refresh-queries", action="store_true",
                    help="ignore queries.json and regenerate (re-bills the calls)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeker-frac", type=float, default=None,
                    help="split the profile pool into disjoint seeker/candidate "
                         "subsets before pairing (real data: ~0.48). Default: off "
                         "(old behavior, every profile is both)")
    ap.add_argument("--max-pairs-per-seeker", type=int, default=None,
                    help="cap labeled pairs kept per seeker after scoring (real "
                         "data: 93 pct of seekers have exactly 1). Default: off "
                         "(no cap)")
    ap.add_argument("--seeker-cap-bump-frac", type=float, default=0.15,
                    help="fraction of seekers (seeded) allowed max+1 pairs "
                         "instead of max, only used with --max-pairs-per-seeker")
    args = ap.parse_args(argv)

    run_pairing(
        profile_run=args.profile_run,
        batch_id=args.batch_id,
        data_dir=args.data_dir,
        artifacts_dir=args.artifacts_dir,
        split_path=args.split_path,
        queries_per_profile=args.queries_per_profile,
        k_per_query=args.k_per_query,
        model_id=args.model_id,
        region=args.region,
        deadband_margin=args.deadband_margin,
        label_mode=args.label_mode,
        pos_frac=args.pos_frac,
        neg_frac=args.neg_frac,
        fusion_mode=args.fusion_mode,
        concurrency=args.concurrency,
        limit=args.limit,
        refresh_queries=args.refresh_queries,
        seed=args.seed,
        seeker_frac=args.seeker_frac,
        max_pairs_per_seeker=args.max_pairs_per_seeker,
        seeker_cap_bump_frac=args.seeker_cap_bump_frac,
    )


if __name__ == "__main__":
    main()
