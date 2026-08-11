#!/usr/bin/env python3
"""Driver for synth_pipeline.pairing_voyage_gemini — Voyage-4-large embeddings,
dense-only recall, gemini-3.1-flash-lite (direct API) judge, over the same
9,659-profile pool already generated for rrf_qwen_full_001.

Isolated copy of scripts/generate_rrf_dataset_qwen_judge.py, simplified: no
profile-generation stage (this pipeline always reuses an existing pool and an
existing queries.json — see synth_pipeline/pairing_voyage_gemini/__init__.py).

    python scripts/generate_voyage_gemini_dataset.py --preset my_run.json
    python scripts/generate_voyage_gemini_dataset.py --skip-judge   # shortlist only, no API spend
    python scripts/generate_voyage_gemini_dataset.py --dry-run

Requires VOYAGE_API_KEY and GEMINI_API_KEY in the environment (.env).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_PRESET = REPO_ROOT / "synth_pipeline" / "pairing_voyage_gemini" / "presets" / "default.json"


def load_preset(path: Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def _clean(section: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in section.items() if not k.startswith("_")}


def run_pairing(preset: dict[str, Any], profile_run: Path, batch_id: str) -> dict[str, Any]:
    from synth_pipeline.config import load_dotenv
    from synth_pipeline.pairing_voyage_gemini.run import RunConfig, run as run_pipeline

    load_dotenv()
    pairing = _clean(preset["pairing"])
    embedding = _clean(preset["embedding"])
    judge = _clean(preset["judge"])

    cfg = RunConfig(
        profile_run=profile_run,
        batch_id=batch_id,
        data_dir=REPO_ROOT / "data",
        artifacts_dir=REPO_ROOT / "artifacts",
        queries_source=REPO_ROOT / preset["queries_source"],
        embed_model=embedding["model"],
        embed_output_dimension=embedding["output_dimension"],
        seeker_frac=pairing["seeker_frac"],
        recall_k=pairing["recall_k"],
        seed=pairing["seed"],
        judge_model=judge["model"],
        judge_concurrency=judge["concurrency"],
        skip_judge=not judge["enabled"],
    )
    return run_pipeline(cfg)


def export(preset: dict[str, Any], batch_id: str, profile_run: Path, summary: dict[str, Any]) -> Path:
    cfg = _clean(preset["export"])
    dest_root = REPO_ROOT / cfg["dest"] / batch_id
    dest_root.mkdir(parents=True, exist_ok=True)
    batch_dir = REPO_ROOT / "artifacts" / "pairing_voyage_gemini" / batch_id

    for name in ("manifest.json", "run_summary.json", "shortlist.json"):
        src = batch_dir / name
        if src.exists():
            shutil.copy2(src, dest_root / name)

    for sub in ("staged", "excluded"):
        src = batch_dir / sub
        if src.is_dir():
            shutil.copytree(src, dest_root / sub, dirs_exist_ok=True)

    if cfg["include_embeddings"]:
        emb = batch_dir / "embeddings"
        if emb.is_dir():
            shutil.copytree(emb, dest_root / "embeddings", dirs_exist_ok=True)
    else:
        emb_manifest = batch_dir / "embeddings" / "manifest.json"
        if emb_manifest.exists():
            shutil.copy2(emb_manifest, dest_root / "embeddings_manifest.json")

    (dest_root / "preset_used.json").write_text(json.dumps(preset, indent=2), encoding="utf-8")
    (dest_root / "README.md").write_text(_readme(batch_id, profile_run, summary), encoding="utf-8")
    print(f"[export] → {dest_root.relative_to(REPO_ROOT)}")
    return dest_root


def _readme(batch_id: str, profile_run: Path, summary: dict[str, Any]) -> str:
    balance = summary.get("balance") or {}
    cost = summary.get("cost") or {}
    return f"""# Synthetic pair batch `{batch_id}` (Voyage-4-large + gemini-3.1-flash-lite)

Produced by `scripts/generate_voyage_gemini_dataset.py`. Settings are in
`preset_used.json` — re-running with that file reproduces this batch.

- Profile pool: `{profile_run.name}` (reused, not regenerated)
- Pairs shortlisted: {summary.get('pairs_shortlisted')}
- Pairs labeled: {summary.get('pairs_labeled')}
- Balance: {balance.get('positive')} positive / {balance.get('negative')} negative
- Density: {balance.get('edges_per_node')} edges per node (real data: 0.673)
- Judge cost: ${cost.get('measured_judge_cost_usd', 0):.4f} (measured, from Gemini token usage x published pricing)

## What's different from rrf_qwen_full_001

Voyage-4-large embeddings (not Qwen3-Embedding-8B), field-restricted embedding
text (seeker: positioning + searchQuery; candidate: positioning + background +
lookingFor), dense-only recall (no BM25/RRF fusion), and gemini-3.1-flash-lite
via the direct Google API with the "focused" judge prompt (not Qwen3-32B on
Bedrock). Queries themselves are reused verbatim from that batch, not
regenerated.

## What these labels are

A model's opinion, not real accept/decline outcomes. **Not promoted** —
nothing here is merged into `data/dataset_*.json`.
"""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--preset", type=Path, default=DEFAULT_PRESET)
    p.add_argument("--batch-id", default=None)
    p.add_argument("--profile-run", type=Path, default=None)
    p.add_argument("--skip-judge", action="store_true")
    p.add_argument("--skip-export", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    preset = load_preset(args.preset)
    batch_id = args.batch_id or preset.get("batch_id") or f"voyage_gemini_{time.strftime('%Y%m%d_%H%M%S')}"

    if args.skip_judge:
        preset["judge"]["enabled"] = False
    if args.skip_export:
        preset["export"]["enabled"] = False
    if args.profile_run:
        preset["profile_run"] = str(args.profile_run)

    print(f"batch_id   {batch_id}")
    print(f"preset     {args.preset}")
    print(f"  judge            {'on' if preset['judge'].get('enabled', True) else 'off'}")
    print(f"  export           {'on' if preset['export'].get('enabled', True) else 'off'}")
    if args.dry_run:
        print(json.dumps(preset, indent=2))
        return 0

    if not preset.get("profile_run"):
        raise SystemExit("no profile_run set in preset — this pipeline never generates its own")
    profile_run = Path(preset["profile_run"])

    summary = run_pairing(preset, profile_run, batch_id)

    if preset["export"]["enabled"]:
        export(preset, batch_id, profile_run, summary)

    print("\n--- done ---")
    print(f"labeled {summary.get('pairs_labeled')} pairs, "
          f"judge cost ${(summary.get('cost') or {}).get('measured_judge_cost_usd', 0):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
