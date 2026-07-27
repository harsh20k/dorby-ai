"""Turn cached LLM-judge verdicts into a pair_id -> [0,1] soft-label file.

Small distillation experiment: instead of training the two-tower LoRA adapter
on hard 0/1 accept/decline labels, use the naive LLM judge's confidence-signed
score (baselines/llm_judge/judge.py::verdict_to_score) as a continuous
ContrastiveLoss target. This script just extracts that mapping from the
judge's verdict cache; twotower/train_distill.py consumes it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from baselines.llm_judge.judge import verdict_to_score


def build_soft_labels(verdicts_path: Path) -> dict[str, float]:
    verdicts = json.loads(verdicts_path.read_text(encoding="utf-8"))
    soft_labels: dict[str, float] = {}
    for key, verdict in verdicts.items():
        pair_id = key.split("|", 1)[0]
        soft_labels[pair_id] = verdict_to_score(verdict)
    return soft_labels


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--verdicts",
        type=Path,
        default=Path(
            "artifacts/llm_judge/openrouter_google_gemini_3_1_flash_lite_naive/verdicts.json"
        ),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/synthetic/judge_soft_labels_naive.json"),
    )
    args = p.parse_args()

    soft_labels = build_soft_labels(args.verdicts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(soft_labels, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(soft_labels)} soft labels -> {args.out}")


if __name__ == "__main__":
    main()
