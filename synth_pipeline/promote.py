"""Promote human-approved staged pairs into dataset_positive/negative.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from synth_pipeline.config import PipelineConfig
from synth_pipeline.nodes.writer import batch_dir, manifest_path
from synth_pipeline.schema import validate_pair_schema


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON list")
    return data


def _existing_keys(rows: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {(r["userContactId"], r["matchContactId"]) for r in rows}


def promote_batch(
    cfg: PipelineConfig,
    batch_id: str,
    *,
    require_human_yes: bool = True,
) -> dict[str, int]:
    staged_dir = batch_dir(cfg, batch_id) / "staged"
    if not staged_dir.exists():
        raise FileNotFoundError(staged_dir)

    pos_path = cfg.positive_path
    neg_path = cfg.negative_path
    positives = _load_dataset(pos_path)
    negatives = _load_dataset(neg_path)
    seen = _existing_keys(positives) | _existing_keys(negatives)

    added = {"pos": 0, "neg": 0, "skipped": 0}
    manifest = json.loads(manifest_path(cfg, batch_id).read_text(encoding="utf-8"))

    for path in sorted(staged_dir.glob("*.json")):
        env = json.loads(path.read_text(encoding="utf-8"))
        review = env.get("human_review")
        if require_human_yes:
            if not isinstance(review, dict) or review.get("verdict") != "yes":
                added["skipped"] += 1
                continue
        pair = env["pair"]
        errors = validate_pair_schema(pair)
        if errors:
            added["skipped"] += 1
            continue
        key = (pair["userContactId"], pair["matchContactId"])
        if key in seen:
            added["skipped"] += 1
            continue
        label = env["label"]
        if label == "pos":
            positives.append(pair)
            added["pos"] += 1
        else:
            negatives.append(pair)
            added["neg"] += 1
        seen.add(key)
        env["status"] = "approved"
        path.write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")

    pos_path.write_text(json.dumps(positives, indent=2) + "\n", encoding="utf-8")
    neg_path.write_text(json.dumps(negatives, indent=2) + "\n", encoding="utf-8")
    manifest["counts"]["approved"] = (
        manifest["counts"].get("approved", 0) + added["pos"] + added["neg"]
    )
    manifest_path(cfg, batch_id).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return added


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-id", required=True)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument(
        "--allow-unreviewed",
        action="store_true",
        help="Promote all staged pairs without human_review.verdict==yes (dev only)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = PipelineConfig()
    if args.data_dir:
        cfg.data_dir = args.data_dir
        cfg.split_path = args.data_dir / "synthetic" / "seed_split.json"
    stats = promote_batch(
        cfg, args.batch_id, require_human_yes=not args.allow_unreviewed
    )
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
