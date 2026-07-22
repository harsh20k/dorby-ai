"""Small-scale test of the knowledge-graph "attribute hub" hypothesis.

Before investing in real KG entity extraction + embedding training
(TransE/RotatE), check cheaply whether shared-attribute connectivity
(industry / stage / location) carries any signal at all on this dataset.
No training, no embeddings: for every holdout pair, count how many of
{industry, stage, location} keyword-tags the seeker and candidate share,
and score pairs with that single integer. If it clears something like
TF-IDF's lexical floor (pair ROC-AUC 0.5922, see docs/baseline-results-
holdout.md), the KG idea is worth building out for real. If it's near
chance, that's a cheap way to kill the idea before spending effort on
extraction + embedding infrastructure.

Usage:
    python scripts/kg_hub_experiment.py --data-dir data
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from baselines.holdout import filter_to_holdout
from baselines.metrics import pair_metrics

# Small hand-curated keyword lexicons — enough to test the hypothesis
# cheaply, not meant to be an exhaustive taxonomy.
INDUSTRY_KEYWORDS = [
    "fintech", "healthcare", "health tech", "biotech", "edtech",
    "e-commerce", "ecommerce", "saas", "artificial intelligence", " ai ",
    "machine learning", "real estate", "logistics", "legal", "venture capital",
    "hospitality", "media", "entertainment", "marketing", "advertising",
    "manufacturing", "climate", "clean energy", "gaming", "crypto",
    "blockchain", "insurance", "retail", "consumer", "b2b", "cybersecurity",
    "agritech", "foodtech", "robotics", "telecom", "nonprofit", "web3",
    "supply chain", "reverse logistics", "wholesale",
]

STAGE_KEYWORDS = [
    "pre-seed", "seed stage", "seed round", "series a", "series b",
    "series c", "series d", "growth stage", "growth-stage", "bootstrapped",
    "early-stage", "early stage", "late-stage", "late stage", "pre-revenue",
    "revenue-generating", "profitable", "ipo", "publicly traded",
]

LOCATION_STOPWORDS = {
    "based", "time", "zone", "eastern", "pacific", "central", "mountain",
    "remote", "onsite", "on-site", "hybrid", "timeline", "constraints",
    "stated", "active", "operating", "network", "best", "suited", "no",
    "explicit", "or", "and", "with", "in", "for", "broader", "united",
    "states", "of", "america", "the", "a", "an", "to", "introductions",
    "sourcing", "flexible", "open", "willing", "travel", "meet", "meetings",
}

_WORD_RE = re.compile(r"[a-z]+")


def _blob(record_file: dict[str, Any]) -> str:
    parts = [
        record_file.get("positioning") or "",
        record_file.get("background") or "",
        record_file.get("lookingFor") or "",
        record_file.get("notes") or "",
    ]
    return " ".join(parts).lower()


def _match_tags(text: str, keywords: list[str]) -> set[str]:
    return {kw for kw in keywords if kw in text}


def _location_tokens(location: str | None) -> set[str]:
    if not location:
        return set()
    tokens = _WORD_RE.findall(location.lower())
    return {t for t in tokens if t not in LOCATION_STOPWORDS and len(t) > 2}


def hub_score(seeker_file: dict[str, Any], candidate_file: dict[str, Any]) -> tuple[int, dict[str, bool]]:
    """Count of shared attribute hubs (industry / stage / location), 0-3."""
    seeker_text = _blob(seeker_file)
    candidate_text = _blob(candidate_file)

    industry_match = bool(
        _match_tags(seeker_text, INDUSTRY_KEYWORDS) & _match_tags(candidate_text, INDUSTRY_KEYWORDS)
    )
    stage_match = bool(
        _match_tags(seeker_text, STAGE_KEYWORDS) & _match_tags(candidate_text, STAGE_KEYWORDS)
    )
    location_match = bool(
        _location_tokens(seeker_file.get("locationAvailability"))
        & _location_tokens(candidate_file.get("locationAvailability"))
    )

    breakdown = {
        "industry": industry_match,
        "stage": stage_match,
        "location": location_match,
    }
    return sum(breakdown.values()), breakdown


def run(data_dir: Path, split_path: Path) -> dict[str, Any]:
    positives = json.loads((data_dir / "dataset_positive.json").read_text())
    negatives = json.loads((data_dir / "dataset_negative.json").read_text())

    positives, negatives = filter_to_holdout(positives, negatives, split_path)
    print(f"holdout: {len(positives)} positives, {len(negatives)} negatives")

    pos_scores = []
    pos_breakdowns = []
    for r in positives:
        s, b = hub_score(r["userContactFile"], r["matchContactFile"])
        pos_scores.append(s)
        pos_breakdowns.append(b)

    neg_scores = []
    neg_breakdowns = []
    for r in negatives:
        s, b = hub_score(r["userContactFile"], r["matchContactFile"])
        neg_scores.append(s)
        neg_breakdowns.append(b)

    pos_scores = np.array(pos_scores, dtype=np.float64)
    neg_scores = np.array(neg_scores, dtype=np.float64)

    pair = pair_metrics(pos_scores, neg_scores)

    def _rate(breakdowns: list[dict[str, bool]], key: str) -> float:
        return float(np.mean([b[key] for b in breakdowns])) if breakdowns else 0.0

    per_category = {
        cat: {
            "pos_match_rate": _rate(pos_breakdowns, cat),
            "neg_match_rate": _rate(neg_breakdowns, cat),
        }
        for cat in ("industry", "stage", "location")
    }

    metrics = {
        "method": "kg_hub_shared_attribute_count",
        "pair": pair,
        "per_category": per_category,
        "n pos": len(positives),
        "n neg": len(negatives),
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--split-path", type=Path, default=None,
        help="defaults to <data-dir>/synthetic/seed_split.json",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/kg_hub_experiment/metrics.json"))
    args = parser.parse_args()

    split_path = args.split_path or (args.data_dir / "synthetic" / "seed_split.json")
    metrics = run(args.data_dir, split_path)

    pair = metrics["pair"]
    print()
    print("=== Pair metrics (shared-attribute-count score) ===")
    print(f"ROC-AUC:              {pair['roc_auc']:.4f}")
    print(f"Average Precision:    {pair['average_precision']:.4f}")
    print(
        f"Best-F1:              {pair['best_f1']:.4f} "
        f"@ threshold={pair['best_f1_threshold']:.4f} "
        f"(acc={pair['best_f1_accuracy']:.4f})"
    )
    print(f"Accuracy @ 0.5:       {pair['accuracy_at_0.5']:.4f}")
    print()
    print("Per-category match rates (pos vs. neg):")
    for cat, rates in metrics["per_category"].items():
        print(f"  {cat:10s}  pos={rates['pos_match_rate']:.3f}  neg={rates['neg_match_rate']:.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
