"""CLI for the RRF pairing pipeline (Qwen3-32B-on-Bedrock judge variant).

Isolated copy of ``synth_pipeline.pairing_rrf`` — see that package's judge.py
docstring for the OpenRouter/flash-lite baseline this is compared against.
This variant swaps the judge for ``qwen.qwen3-32b-v1:0`` via AWS Bedrock,
~3x cheaper per call but lower measured AUC (see judge.py here).

    export AWS_PROFILE=tf_provisioner AWS_DEFAULT_REGION=us-east-1
    python -m synth_pipeline.pairing_rrf_qwen_judge \
      --profile-run artifacts/bedrock_synth/run_<ts> \
      --batch-id rrf_qwen_001 \
      --data-dir /Users/harsh/Artifacts/dorby-ai/data

Add ``--skip-judge`` to run everything up to and including fusion without
spending anything on Bedrock — the shortlists are written either way, so this
is the cheap way to inspect what *would* be judged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from synth_pipeline.config import load_dotenv
from synth_pipeline.pairing_rrf_qwen_judge import embed as embed_mod
from synth_pipeline.pairing_rrf_qwen_judge import fuse as fuse_mod
from synth_pipeline.pairing_rrf_qwen_judge.run import DEFAULT_BEDROCK_MODEL, RunConfig, run

REPO_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m synth_pipeline.pairing_rrf_qwen_judge")
    p.add_argument("--profile-run", type=Path, required=True,
                   help="a bedrock_synth / local_gemma_synth run directory")
    p.add_argument("--batch-id", required=True)
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data",
                   help="data/ is gitignored — pass explicitly when running from a worktree")
    p.add_argument("--artifacts-dir", type=Path, default=REPO_ROOT / "artifacts")

    p.add_argument("--embed-model", default=embed_mod.DEFAULT_MODEL)
    p.add_argument("--embed-batch-size", type=int, default=2)
    p.add_argument("--embed-max-length", type=int, default=4096)
    p.add_argument("--embed-device", default=None, help="cuda | mps | cpu (default: auto)")
    p.add_argument("--embed-backend", choices=("local", "modal"), default="local",
                   help="modal runs on an A100 — required for any 7-8B model")

    p.add_argument("--bedrock-model", default=DEFAULT_BEDROCK_MODEL)
    p.add_argument("--region", default="us-east-1")

    p.add_argument("--seeker-frac", type=float, default=0.43)
    p.add_argument("--recall-k", type=int, default=10, help="per channel, before fusion")
    p.add_argument("--top-k", type=int, default=5, help="shortlist depth sent to the judge")
    p.add_argument("--max-pairs-per-seeker", type=int, default=None,
                   help="cap judged pairs per seeker across all their sections")
    p.add_argument("--min-dense-similarity", type=float, default=None,
                   help="absolute cosine floor; off by default (needs real-pair calibration)")
    p.add_argument("--dense-weight", type=float, default=fuse_mod.DENSE_WEIGHT)
    p.add_argument("--lexical-weight", type=float, default=fuse_mod.LEXICAL_WEIGHT)
    p.add_argument("--rrf-k", type=int, default=fuse_mod.RRF_K)

    p.add_argument("--judge-concurrency", type=int, default=4)
    p.add_argument("--allow-duplicate-pairs", action="store_true",
                   help="keep the same (seeker,candidate) under several queries; "
                        "risks contradictory labels since the schema has no query identity")
    p.add_argument("--skip-judge", action="store_true",
                   help="stop after fusion; no OpenRouter spend")
    p.add_argument("--no-chroma", action="store_true",
                   help="use the exact NumPy store instead of Chroma")
    p.add_argument("--limit", type=int, default=None, help="cap profiles loaded (smoke tests)")
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    cfg = RunConfig(
        profile_run=args.profile_run,
        batch_id=args.batch_id,
        data_dir=args.data_dir,
        artifacts_dir=args.artifacts_dir,
        embed_model=args.embed_model,
        bedrock_model=args.bedrock_model,
        region=args.region,
        seeker_frac=args.seeker_frac,
        recall_k=args.recall_k,
        top_k=args.top_k,
        max_pairs_per_seeker=args.max_pairs_per_seeker,
        min_dense_similarity=args.min_dense_similarity,
        dense_weight=args.dense_weight,
        lexical_weight=args.lexical_weight,
        rrf_k=args.rrf_k,
        seed=args.seed,
        limit=args.limit,
        judge_concurrency=args.judge_concurrency,
        use_chroma=not args.no_chroma,
        embed_batch_size=args.embed_batch_size,
        embed_max_length=args.embed_max_length,
        embed_device=args.embed_device,
        embed_backend=args.embed_backend,
        dedupe_pairs=not args.allow_duplicate_pairs,
        skip_judge=args.skip_judge,
    )
    summary = run(cfg)
    print("\n--- run summary ---")
    print(f"batch          {summary['batch_id']}")
    print(f"shortlisted    {summary['pairs_shortlisted']}")
    print(f"labeled        {summary['pairs_labeled']}")
    if summary.get("balance"):
        b = summary["balance"]
        print(f"balance        {b['positive']} pos / {b['negative']} neg "
              f"({b['positive_frac']:.1%} positive)")
        print(f"density        {b['edges_per_node']} edges/node "
              f"(real reference {b['real_reference']['edges_per_node']})")
    judge_cost = summary["cost"].get("measured_judge_cost_usd") or 0.0
    print(f"judge cost     ${judge_cost:.4f} (measured, not estimated)")
    bq = summary["cost"]["bedrock_query_generation"]
    print(f"bedrock tokens {bq['total_tokens']} over {bq['calls']} calls")
    print(f"wall clock     {summary['wall_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
