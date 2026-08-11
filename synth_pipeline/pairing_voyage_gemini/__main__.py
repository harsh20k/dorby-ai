"""CLI for the Voyage-4-large + Gemini pairing pipeline.

Isolated copy of ``synth_pipeline.pairing_rrf_qwen_judge`` — see this
package's ``__init__.py`` for what's different and why.

    python -m synth_pipeline.pairing_voyage_gemini \
      --profile-run artifacts/bedrock_synth/run_20260804_023936 \
      --batch-id voyage_gemini_001 \
      --data-dir /Users/harsh/Artifacts/dorby-ai/data

Add ``--skip-judge`` to run everything up to and including dedupe without
spending anything on the Gemini API.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from synth_pipeline.config import load_dotenv
from synth_pipeline.pairing_voyage_gemini import embed as embed_mod
from synth_pipeline.pairing_voyage_gemini.judge import DEFAULT_MODEL as DEFAULT_JUDGE_MODEL
from synth_pipeline.pairing_voyage_gemini.run import DEFAULT_QUERIES_SOURCE, RunConfig, run

REPO_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m synth_pipeline.pairing_voyage_gemini")
    p.add_argument("--profile-run", type=Path, required=True,
                   help="a bedrock_synth / local_gemma_synth run directory")
    p.add_argument("--batch-id", required=True)
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data",
                   help="data/ is gitignored — pass explicitly when running from a worktree")
    p.add_argument("--artifacts-dir", type=Path, default=REPO_ROOT / "artifacts")
    p.add_argument("--queries-source", type=Path, default=DEFAULT_QUERIES_SOURCE,
                   help="reused queries.json from a prior batch over the same profile pool")

    p.add_argument("--embed-model", default=embed_mod.DEFAULT_MODEL)
    p.add_argument("--embed-output-dimension", type=int, default=embed_mod.DEFAULT_OUTPUT_DIMENSION)

    p.add_argument("--seeker-frac", type=float, default=0.43)
    p.add_argument("--recall-k", type=int, default=10)

    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    p.add_argument("--judge-concurrency", type=int, default=4)
    p.add_argument("--allow-duplicate-pairs", action="store_true",
                   help="keep the same (seeker,candidate) under several queries")
    p.add_argument("--skip-judge", action="store_true",
                   help="stop after dedupe; no Gemini API spend")
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
        queries_source=args.queries_source,
        embed_model=args.embed_model,
        embed_output_dimension=args.embed_output_dimension,
        seeker_frac=args.seeker_frac,
        recall_k=args.recall_k,
        seed=args.seed,
        limit=args.limit,
        judge_model=args.judge_model,
        judge_concurrency=args.judge_concurrency,
        use_chroma=not args.no_chroma,
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
    print(f"wall clock     {summary['wall_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
