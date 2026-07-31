"""Aggregate every all-200 result into one comparison table.

Counterpart to ``scripts/export_baseline_results.py``, which builds the 69-pair
holdout table. Same idea, different population — and the population is the whole
point: the holdout table ranks models on 29 positive queries against a
65-candidate pool, which has now reversed a conclusion three times in this
project. This one ranks them on 100 positive queries against 178 candidates.

Writes ``docs/baseline-results-real200.{json,md}``. Rows come from whatever
exists under ``artifacts/eval_real_full/``; nothing is recomputed here.

    python -m eval_real_full.export
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "eval_real_full"
OUT_JSON = ROOT / "docs" / "baseline-results-real200.json"
OUT_MD = ROOT / "docs" / "baseline-results-real200.md"

# label -> (metrics path, display name, family)
# `family` groups rows in the table; it is not used for ranking.
SOURCES: dict[str, tuple[str, str, str]] = {
    "tfidf": ("real200_baselines_local/tfidf/metrics.json", "TF-IDF (lexical)", "baseline"),
    "bert": ("real200_baselines/bert/metrics.json", "Frozen BERT", "baseline"),
    "frozen": ("real200_001/frozen/metrics.json", "Voyage-4-nano (frozen)", "baseline"),
    "voyage_large": (
        "real200_voyage_large/metrics.json",
        "Voyage-4-large (production)",
        "baseline",
    ),
    "qwen8b": (
        "real200_baselines/qwen8b/metrics.json",
        "Qwen3-Embedding-8B (open)",
        "open-weight",
    ),
    "e5_mistral": (
        "real200_baselines/e5_mistral/metrics.json",
        "E5-Mistral-7B-instruct (open)",
        "open-weight",
    ),
    "zembed": (
        "real200_baselines/zembed/metrics.json",
        "zembed-1-embedding (open)",
        "open-weight",
    ),
    "bge_en_icl": (
        "real200_baselines/bge_en_icl/metrics.json",
        "BGE-en-ICL (open, zero-shot)",
        "open-weight",
    ),
    "nv_embed": (
        "real200_baselines/nv_embed/metrics.json",
        "NV-Embed-v2 (open, non-commercial, approx.)",
        "open-weight",
    ),
    "arm_a_v2": ("real200_001/arm_a_v2/metrics.json", "twotower Arm A (v2)", "fine-tuned"),
    "top1_ctrl": (
        "real200_top1/top1_ctrl/metrics.json",
        "twotower top1_ctrl",
        "fine-tuned",
    ),
    "top1_sharp": (
        "real200_top1/top1_sharp/metrics.json",
        "twotower top1_sharp",
        "fine-tuned",
    ),
    "qwen_micro6": (
        "real200_qwen/qwen_micro6/metrics.json",
        "twotower Qwen micro-6",
        "fine-tuned",
    ),
    "qwen_micro1": (
        "real200_qwen/qwen_micro1/metrics.json",
        "twotower Qwen micro-1",
        "fine-tuned",
    ),
}

METRIC_COLUMNS = (
    ("pair_auc", "pair AUC"),
    ("hard_neg_auc", "hard-neg AUC"),
    ("easy_neg_auc", "easy-neg AUC"),
    ("mrr", "MRR"),
    ("recall@1", "R@1"),
    ("recall@10", "R@10"),
)


def _row(metrics: dict, subset: str) -> dict | None:
    sub = metrics.get("subsets", {}).get(subset)
    if not sub:
        return None
    hardness = sub.get("slices", {}).get("neg_hardness", {})
    return {
        "pair_auc": sub["pair"]["roc_auc"],
        "hard_neg_auc": (hardness.get("hard") or {}).get("pair_auc"),
        "easy_neg_auc": (hardness.get("easy") or {}).get("pair_auc"),
        "mrr": sub["retrieval"]["mrr"],
        "recall@1": sub["retrieval"]["recall@1"],
        "recall@10": sub["retrieval"]["recall@10"],
        "n_candidates": sub.get("n_candidates"),
    }


def collect() -> dict:
    rows: dict[str, dict] = {}
    missing: list[str] = []
    for key, (rel, label, family) in SOURCES.items():
        path = ARTIFACTS / rel
        if not path.is_file():
            missing.append(key)
            continue
        metrics = json.loads(path.read_text())
        entry = {
            "label": label,
            "family": family,
            "source": str(path.relative_to(ROOT)),
            "library_versions": metrics.get("library_versions"),
            "subsets": {},
        }
        for subset in ("all", "train", "holdout"):
            row = _row(metrics, subset)
            if row:
                entry["subsets"][subset] = row
        rows[key] = entry
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "population": "the 200 real seed pairs (100 positive queries, 178-candidate corpus)",
        "rows": rows,
        "missing": missing,
    }


def _fmt(v: object) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def build_md(payload: dict) -> str:
    rows = payload["rows"]
    lines: list[str] = [
        "# All-200 real-pair comparison",
        "",
        "Every model in this project scored on the **same 200 real pairs** — 100",
        "positive queries ranked against a 178-candidate corpus — rather than the",
        "69-pair holdout used by `docs/baseline-results-holdout.md`.",
        "",
        "Retrieval metrics are comparable **between models within one subset only**.",
        "A larger candidate pool is strictly harder, so `all` (178 candidates) and",
        "`holdout` (65) must never be compared to each other.",
        "",
    ]

    for subset, title in (("all", "All 200 pairs"), ("holdout", "Holdout 69 pairs (reference)")):
        present = [(k, v) for k, v in rows.items() if subset in v["subsets"]]
        if not present:
            continue
        present.sort(key=lambda kv: kv[1]["subsets"][subset]["mrr"] or 0, reverse=True)
        n_cand = present[0][1]["subsets"][subset]["n_candidates"]
        lines += [
            f"## {title} — corpus {n_cand} candidates",
            "",
            "Ranked by MRR.",
            "",
            "| model | family | " + " | ".join(c[1] for c in METRIC_COLUMNS) + " |",
            "|---|---|" + "---|" * len(METRIC_COLUMNS),
        ]
        for key, entry in present:
            r = entry["subsets"][subset]
            cells = " | ".join(_fmt(r[c[0]]) for c in METRIC_COLUMNS)
            lines.append(f"| {entry['label']} | {entry['family']} | {cells} |")
        lines.append("")

    if payload["missing"]:
        lines += [
            "## Not yet scored",
            "",
            ", ".join(f"`{m}`" for m in payload["missing"]),
            "",
        ]
    lines += [
        "---",
        "",
        f"Generated {payload['generated_at']} by `python -m eval_real_full.export`.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    payload = collect()
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    OUT_MD.write_text(build_md(payload))
    print(f"wrote {OUT_JSON.relative_to(ROOT)}")
    print(f"wrote {OUT_MD.relative_to(ROOT)}")
    if payload["missing"]:
        print("missing (not yet run): " + ", ".join(payload["missing"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
