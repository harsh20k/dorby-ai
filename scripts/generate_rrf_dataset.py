#!/usr/bin/env python3
"""One command for a whole synthetic dataset: generate → pair → judge → export.

This is the reusable template. Everything tunable lives in a preset JSON file
rather than in flags, so producing another dataset is "copy the preset, change a
number, run it" — and the preset that produced any batch is copied into that
batch's output directory, so a result can always be traced back and reproduced.

    # as-is, using the shipped defaults
    python scripts/generate_rrf_dataset.py

    # tuned
    cp synth_pipeline/pairing_rrf/presets/default.json my_run.json
    $EDITOR my_run.json
    python scripts/generate_rrf_dataset.py --preset my_run.json

    # re-pair a pool you already paid to generate
    python scripts/generate_rrf_dataset.py --preset my_run.json \
        --profile-run artifacts/bedrock_synth/run_<ts> --skip-generate

Stages are independently skippable and each writes before the next starts, so a
failure late in the chain never costs the expensive artifacts earlier in it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_PRESET = REPO_ROOT / "synth_pipeline" / "pairing_rrf" / "presets" / "default.json"


def load_preset(path: Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    # Keys prefixed with "_" are documentation, not settings.
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def _clean(section: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in section.items() if not k.startswith("_")}


def generate_profiles(cfg: dict[str, Any], *, log_path: Path) -> Path:
    """Run the Bedrock profile generator; return the run directory it created."""
    before = set((REPO_ROOT / "artifacts" / "bedrock_synth").glob("run_*")) if (
        REPO_ROOT / "artifacts" / "bedrock_synth"
    ).exists() else set()

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "bedrock_profile_gen.py"),
        "--max-profiles", str(cfg["count"]),
        "--concurrency", str(cfg["concurrency"]),
        "--model-id", cfg["model_id"],
        "--region", cfg["region"],
    ]
    print(f"[generate] {' '.join(cmd)}")
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO_ROOT)
    if proc.returncode != 0:
        raise SystemExit(f"profile generation failed (rc={proc.returncode}); see {log_path}")

    after = set((REPO_ROOT / "artifacts" / "bedrock_synth").glob("run_*"))
    new = sorted(after - before)
    if not new:
        raise SystemExit("profile generation produced no new run directory")
    run_dir = new[-1]
    n_ok = len(list((run_dir / "profiles").glob("*.json")))
    print(f"[generate] {n_ok} profiles in {run_dir.name}")
    if n_ok == 0:
        raise SystemExit(
            f"generation wrote 0 profiles — check {log_path}. A KeyError on every "
            "profile usually means PROFILE_GEN_PROMPT_GENERATE is unpinned and fell "
            "back to the v1 prompt."
        )
    return run_dir


def run_pairing(preset: dict[str, Any], profile_run: Path, batch_id: str) -> dict[str, Any]:
    from synth_pipeline.config import load_dotenv
    from synth_pipeline.pairing_rrf.run import RunConfig, run as run_pipeline

    load_dotenv()
    pairing = _clean(preset["pairing"])
    embedding = _clean(preset["embedding"])
    judge = _clean(preset["judge"])

    cfg = RunConfig(
        profile_run=profile_run,
        batch_id=batch_id,
        data_dir=REPO_ROOT / "data",
        artifacts_dir=REPO_ROOT / "artifacts",
        embed_model=embedding["model"],
        embed_backend=embedding["backend"],
        embed_batch_size=embedding["batch_size"],
        embed_max_length=embedding["max_length"],
        use_chroma=embedding["use_chroma"],
        bedrock_model=preset["generate_profiles"]["model_id"],
        region=preset["generate_profiles"]["region"],
        seeker_frac=pairing["seeker_frac"],
        recall_k=pairing["recall_k"],
        top_k=pairing["top_k"],
        dense_weight=pairing["dense_weight"],
        lexical_weight=pairing["lexical_weight"],
        rrf_k=pairing["rrf_k"],
        max_pairs_per_seeker=pairing["max_pairs_per_seeker"],
        min_dense_similarity=pairing["min_dense_similarity"],
        seed=pairing["seed"],
        judge_concurrency=judge["concurrency"],
        skip_judge=not judge["enabled"],
    )
    return run_pipeline(cfg)


def export(
    preset: dict[str, Any],
    batch_id: str,
    profile_run: Path,
    summary: dict[str, Any],
) -> Path:
    """Copy the run's durable outputs somewhere git actually tracks.

    ``artifacts/`` is gitignored, which is exactly how the previous 100-profile
    batch was lost. Labels, manifest, queries, shortlists and the preset always
    go; embeddings only when asked, since they are large.
    """
    cfg = _clean(preset["export"])
    dest_root = REPO_ROOT / cfg["dest"] / batch_id
    dest_root.mkdir(parents=True, exist_ok=True)
    batch_dir = REPO_ROOT / "artifacts" / "pairing_rrf" / batch_id

    for name in ("manifest.json", "run_summary.json", "queries.json", "shortlists.json"):
        src = batch_dir / name
        if src.exists():
            shutil.copy2(src, dest_root / name)

    for sub in ("staged", "excluded"):
        src = batch_dir / sub
        if src.is_dir():
            shutil.copytree(src, dest_root / sub, dirs_exist_ok=True)

    profiles_src = profile_run / "profiles"
    if profiles_src.is_dir():
        shutil.copytree(profiles_src, dest_root / "profiles", dirs_exist_ok=True)
    for name in ("manifest.jsonl",):
        src = profile_run / name
        if src.exists():
            shutil.copy2(src, dest_root / f"profile_gen_{name}")

    if cfg["include_embeddings"]:
        emb = batch_dir / "embeddings"
        if emb.is_dir():
            shutil.copytree(emb, dest_root / "embeddings", dirs_exist_ok=True)
    else:
        emb_manifest = batch_dir / "embeddings" / "manifest.json"
        if emb_manifest.exists():
            shutil.copy2(emb_manifest, dest_root / "embeddings_manifest.json")

    (dest_root / "preset_used.json").write_text(
        json.dumps(preset, indent=2), encoding="utf-8"
    )
    (dest_root / "README.md").write_text(_readme(batch_id, profile_run, summary), encoding="utf-8")
    print(f"[export] → {dest_root.relative_to(REPO_ROOT)}")
    return dest_root


def _readme(batch_id: str, profile_run: Path, summary: dict[str, Any]) -> str:
    balance = summary.get("balance") or {}
    cost = summary.get("cost") or {}
    return f"""# Synthetic pair batch `{batch_id}`

Produced by `scripts/generate_rrf_dataset.py`. The exact settings are in
`preset_used.json` — re-running with that file reproduces this batch.

- Profile pool: `{profile_run.name}`
- Pairs shortlisted: {summary.get('pairs_shortlisted')}
- Pairs labeled: {summary.get('pairs_labeled')}
- Balance: {balance.get('positive')} positive / {balance.get('negative')} negative
- Density: {balance.get('edges_per_node')} edges per node (real data: 0.673)
- Judge cost: ${cost.get('measured_judge_cost_usd', 0):.4f} (measured, from OpenRouter usage)

## What these labels are

A model's opinion, not real accept/decline outcomes. The judge
(`google/gemini-3.1-flash-lite`, naive framing) measured 0.6358 pair AUC and
0.5942 decision accuracy on the real 69-pair holdout. There is no deadband:
every judged pair is labeled, so borderline pairs carry some wrong labels.

Negatives are the more interesting half — each was surfaced by both a dense
embedding channel and BM25 and still refused by the judge, which is closer to
the real data's "production recommended it, a human declined" than any
constructed mismatch.

**Not promoted.** Nothing here is merged into `data/dataset_*.json`.
"""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--preset", type=Path, default=DEFAULT_PRESET)
    p.add_argument("--batch-id", default=None, help="overrides the preset")
    p.add_argument("--profile-run", type=Path, default=None,
                   help="reuse an existing profile pool instead of generating one")
    p.add_argument("--skip-generate", action="store_true")
    p.add_argument("--skip-judge", action="store_true")
    p.add_argument("--skip-export", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="print the resolved plan and stop")
    args = p.parse_args(argv)

    preset = load_preset(args.preset)
    batch_id = args.batch_id or preset.get("batch_id") or f"rrf_{time.strftime('%Y%m%d_%H%M%S')}"

    if args.skip_generate:
        preset["generate_profiles"]["enabled"] = False
    if args.skip_judge:
        preset["judge"]["enabled"] = False
    if args.skip_export:
        preset["export"]["enabled"] = False
    if args.profile_run:
        preset["profile_run"] = str(args.profile_run)

    print(f"batch_id   {batch_id}")
    print(f"preset     {args.preset}")
    for stage in ("generate_profiles", "pairing", "judge", "export"):
        print(f"  {stage:<18} {'on' if preset[stage].get('enabled', True) else 'off'}")
    if args.dry_run:
        print(json.dumps(preset, indent=2))
        return 0

    logs = REPO_ROOT / "artifacts" / "pairing_rrf" / batch_id
    logs.mkdir(parents=True, exist_ok=True)

    if preset["generate_profiles"]["enabled"]:
        profile_run = generate_profiles(
            _clean(preset["generate_profiles"]), log_path=logs / "profile_gen.log"
        )
    elif preset.get("profile_run"):
        profile_run = Path(preset["profile_run"])
    else:
        raise SystemExit("generation is off and no profile_run is set — nothing to pair")

    if not preset["pairing"]["enabled"]:
        print("pairing disabled; stopping after generation")
        return 0

    summary = run_pairing(preset, profile_run, batch_id)

    if preset["export"]["enabled"]:
        export(preset, batch_id, profile_run, summary)

    print("\n--- done ---")
    print(f"labeled {summary.get('pairs_labeled')} pairs, "
          f"judge cost ${(summary.get('cost') or {}).get('measured_judge_cost_usd', 0):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
