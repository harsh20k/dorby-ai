#!/usr/bin/env python3
"""Merge artifacts/*/metrics.json into docs/baseline-results-all.{json,md}."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
OUT_JSON = ROOT / "docs" / "baseline-results-all.json"
OUT_MD = ROOT / "docs" / "baseline-results-all.md"

# with-query (searchQuery on seeker) then profile-only / no-query siblings
BASELINES = (
    "tfidf",
    "bert_frozen",
    "bert_frozen_no_query",
    "voyage_nano",
    "voyage_nano_no_query",
    "voyage_large",
    "voyage_large_no_query",
)
LABELS = {
    "tfidf": "TF-IDF (lexical)",
    "bert_frozen": "Frozen BERT",
    "bert_frozen_no_query": "Frozen BERT (no query)",
    "voyage_nano": "Voyage-4-nano",
    "voyage_nano_no_query": "Voyage-4-nano (no query)",
    "voyage_large": "Voyage-4-large",
    "voyage_large_no_query": "Voyage-4-large (no query)",
}
PROTOCOL_NOTE = (
    "same pair/retrieval/slices for all; "
    "(no query) = profile-only seeker text (no searchQuery)"
)


def _fmt(v: object, digits: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int) and not isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        if v != v:  # NaN
            return "—"
        return f"{v:.{digits}f}"
    return str(v)


def _pct(d: dict | None, side: str, key: str) -> object:
    if not d:
        return None
    return (d.get(side) or {}).get(key)


def load_baselines() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for name in BASELINES:
        path = ARTIFACTS / name / "metrics.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}")
        out[name] = json.loads(path.read_text())
    return out


def load_from_paths(paths: dict[str, Path]) -> dict[str, dict]:
    """Like load_baselines() but for an explicit {name: path} mapping —
    used for the holdout comparison, where twotower's metrics_holdout.json
    lives at a different path convention than artifacts/<name>/metrics.json."""
    out: dict[str, dict] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}")
        out[name] = json.loads(path.read_text())
    return out


def build_json(baselines: dict[str, dict], protocol_note: str = PROTOCOL_NOTE) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "protocol_note": protocol_note,
        "baselines": baselines,
    }


def build_md(
    payload: dict,
    names: list[str] | None = None,
    labels: dict[str, str] | None = None,
    title: str = "# Baseline results (all metrics)",
    sources_note: str = (
        "Sources: `artifacts/{bert_frozen,bert_frozen_no_query,voyage_nano,"
        "voyage_nano_no_query,voyage_large,voyage_large_no_query}/metrics.json`."
    ),
) -> str:
    baselines: dict[str, dict] = payload["baselines"]
    names = names if names is not None else list(BASELINES)
    labels = labels if labels is not None else LABELS
    cols = [labels[n] for n in names]

    def col_vals(getter) -> list[str]:
        return [_fmt(getter(baselines[n])) for n in names]

    lines: list[str] = [
        title,
        "",
        f"Generated: `{payload['generated_at']}`",
        "",
        f"Protocol: {payload['protocol_note']}. {sources_note}",
        "",
        "Metric definitions: [baseline-metrics.md](baseline-metrics.md).",
        "",
        "Refresh:",
        "```bash",
        "python scripts/export_baseline_results.py",
        "```",
        "",
        "## Pair",
        "",
        "| Metric | " + " | ".join(cols) + " |",
        "|--------|" + "|".join(["--------"] * len(cols)) + "|",
    ]

    pair_rows = [
        ("ROC-AUC", lambda m: m["pair"].get("roc_auc")),
        ("Average precision", lambda m: m["pair"].get("average_precision")),
        ("Best F1", lambda m: m["pair"].get("best_f1")),
        ("Best-F1 threshold", lambda m: m["pair"].get("best_f1_threshold")),
        ("Accuracy @ best-F1", lambda m: m["pair"].get("best_f1_accuracy")),
        ("Accuracy @ 0.5", lambda m: m["pair"].get("accuracy_at_0.5")),
        ("Mean cos (pos)", lambda m: m["pair"].get("mean_cosine_positive")),
        ("Mean cos (neg)", lambda m: m["pair"].get("mean_cosine_negative")),
        ("Mean cos gap", lambda m: m["pair"].get("mean_cosine_gap")),
        ("Std cos (pos)", lambda m: m["pair"].get("std_cosine_positive")),
        ("Std cos (neg)", lambda m: m["pair"].get("std_cosine_negative")),
        (
            "Pos p10 / p50 / p90",
            lambda m: (
                f"{_fmt(_pct(m['pair'].get('score_percentiles'), 'positive', 'p10'))} / "
                f"{_fmt(_pct(m['pair'].get('score_percentiles'), 'positive', 'p50'))} / "
                f"{_fmt(_pct(m['pair'].get('score_percentiles'), 'positive', 'p90'))}"
                if m["pair"].get("score_percentiles")
                else "—"
            ),
        ),
        (
            "Neg p10 / p50 / p90",
            lambda m: (
                f"{_fmt(_pct(m['pair'].get('score_percentiles'), 'negative', 'p10'))} / "
                f"{_fmt(_pct(m['pair'].get('score_percentiles'), 'negative', 'p50'))} / "
                f"{_fmt(_pct(m['pair'].get('score_percentiles'), 'negative', 'p90'))}"
                if m["pair"].get("score_percentiles")
                else "—"
            ),
        ),
        ("n pos / n neg", lambda m: f"{m['pair'].get('num_positive')} / {m['pair'].get('num_negative')}"),
    ]
    for label, getter in pair_rows:
        lines.append(f"| {label} | " + " | ".join(col_vals(getter)) + " |")

    lines += [
        "",
        "## Retrieval",
        "",
        "| Metric | " + " | ".join(cols) + " |",
        "|--------|" + "|".join(["--------"] * len(cols)) + "|",
    ]
    ret_rows = [
        ("MRR", lambda m: m["retrieval"].get("mrr")),
        ("MAP", lambda m: m["retrieval"].get("map")),
        ("Mean rank", lambda m: m["retrieval"].get("mean_rank")),
        ("Median rank", lambda m: m["retrieval"].get("median_rank")),
        ("Top-1", lambda m: m["retrieval"].get("top1")),
        ("R@1", lambda m: m["retrieval"].get("recall@1")),
        ("R@5", lambda m: m["retrieval"].get("recall@5")),
        ("R@10", lambda m: m["retrieval"].get("recall@10")),
        ("NDCG@1", lambda m: m["retrieval"].get("ndcg@1")),
        ("NDCG@5", lambda m: m["retrieval"].get("ndcg@5")),
        ("NDCG@10", lambda m: m["retrieval"].get("ndcg@10")),
        ("P@1", lambda m: m["retrieval"].get("precision@1")),
        ("P@5", lambda m: m["retrieval"].get("precision@5")),
        ("P@10", lambda m: m["retrieval"].get("precision@10")),
        ("n queries", lambda m: m["retrieval"].get("num_queries")),
    ]
    for label, getter in ret_rows:
        lines.append(f"| {label} | " + " | ".join(col_vals(getter)) + " |")

    lines += [
        "",
        "## Slices: neg hardness (pair AUC)",
        "",
        "| Slice | " + " | ".join(cols) + " |",
        "|-------|" + "|".join(["--------"] * len(cols)) + "|",
    ]
    for slice_name in ("easy", "hard"):
        lines.append(
            f"| {slice_name} AUC | "
            + " | ".join(
                col_vals(
                    lambda m, s=slice_name: (m.get("slices") or {})
                    .get("neg_hardness", {})
                    .get(s, {})
                    .get("pair_auc")
                )
            )
            + " |"
        )
        lines.append(
            f"| {slice_name} n_neg | "
            + " | ".join(
                col_vals(
                    lambda m, s=slice_name: (m.get("slices") or {})
                    .get("neg_hardness", {})
                    .get(s, {})
                    .get("n_negatives")
                )
            )
            + " |"
        )

    # Intent union across baselines
    intent_keys: set[str] = set()
    for n in names:
        intent_keys |= set(((baselines[n].get("slices") or {}).get("intent") or {}).keys())
    intent_order = sorted(intent_keys)

    lines += [
        "",
        "## Slices: intent breakdown",
        "",
        "| Intent | Metric | " + " | ".join(cols) + " |",
        "|--------|--------|" + "|".join(["--------"] * len(cols)) + "|",
    ]
    for intent in intent_order:
        for metric_label, key in (
            ("n_pairs", "n_pairs"),
            ("n_queries", "n_queries"),
            ("AUC", "pair_auc"),
            ("MRR", "mrr"),
            ("R@10", "recall@10"),
        ):
            vals = []
            for n in names:
                block = ((baselines[n].get("slices") or {}).get("intent") or {}).get(intent) or {}
                vals.append(_fmt(block.get(key)))
            lines.append(f"| {intent} | {metric_label} | " + " | ".join(vals) + " |")

    lines += [
        "",
        "## Model metadata",
        "",
        "| Field | " + " | ".join(cols) + " |",
        "|-------|" + "|".join(["--------"] * len(cols)) + "|",
        "| model_name | "
        + " | ".join(col_vals(lambda m: m.get("model_name")))
        + " |",
        "| device | " + " | ".join(col_vals(lambda m: m.get("device", "—"))) + " |",
        "| max_length | " + " | ".join(col_vals(lambda m: m.get("max_length", "—"))) + " |",
        "| output / truncate dim | "
        + " | ".join(
            col_vals(lambda m: m.get("output_dimension") or m.get("truncate_dim") or "—")
        )
        + " |",
        "",
    ]
    return "\n".join(lines)


# Matched-population comparison on the frozen 69-pair real holdout only —
# see docs/possible-bugs.md #3. Baseline entries come from the *_holdout
# artifact dirs (produced by `eval.py --holdout-only`); twotower entries are
# named per run_id and point at that run's metrics_holdout.json directly,
# since twotower doesn't share the artifacts/<name>/metrics.json convention.
HOLDOUT_OUT_JSON = ROOT / "docs" / "baseline-results-holdout.json"
HOLDOUT_OUT_MD = ROOT / "docs" / "baseline-results-holdout.md"
HOLDOUT_PATHS = {
    "tfidf": ARTIFACTS / "tfidf_holdout" / "metrics.json",
    "bert_frozen": ARTIFACTS / "bert_frozen_holdout" / "metrics.json",
    "voyage_nano": ARTIFACTS / "voyage_nano_holdout" / "metrics.json",
    "voyage_large": ARTIFACTS / "voyage_large_holdout" / "metrics.json",
    "hybrid_tfidf_voyage": ARTIFACTS / "hybrid_tfidf_voyage_holdout" / "metrics.json",
    "hf_embedding_qwen_qwen3-embedding-8b": ARTIFACTS
    / "hf_embedding_qwen_qwen3-embedding-8b"
    / "metrics.json",
}
HOLDOUT_LABELS = {
    "tfidf": "TF-IDF (lexical)",
    "bert_frozen": "Frozen BERT",
    "voyage_nano": "Voyage-4-nano",
    "voyage_large": "Voyage-4-large (prod)",
    "hybrid_tfidf_voyage": "Hybrid TF-IDF+nano",
    "hf_embedding_qwen_qwen3-embedding-8b": "Qwen3-Embedding-8B (open, Modal)",
}
HOLDOUT_PROTOCOL_NOTE = (
    "frozen 69-pair real holdout (data/synthetic/seed_split.json eval_pair_ids) only — "
    "same population for every row, unlike baseline-results-all.md's full-dataset numbers. "
    "See docs/possible-bugs.md #3 and docs/twotower-run-001-results.md."
)


def add_twotower_holdout_run(run_id: str, metrics_path: Path | None = None) -> None:
    """Register a twotower run's metrics_holdout.json for the holdout comparison."""
    path = metrics_path or (ARTIFACTS / "twotower" / f"{run_id}_holdout_eval" / "metrics_holdout.json")
    HOLDOUT_PATHS[f"twotower_{run_id}"] = path
    HOLDOUT_LABELS[f"twotower_{run_id}"] = f"twotower {run_id}"


def build_holdout_comparison() -> None:
    # Required core baselines; hybrid + twotower are optional extras.
    required = ("tfidf", "bert_frozen", "voyage_nano", "voyage_large")
    missing_required = [n for n in required if not HOLDOUT_PATHS[n].is_file()]
    if missing_required:
        print(
            "skipping holdout comparison — missing required: "
            + ", ".join(str(HOLDOUT_PATHS[n]) for n in missing_required)
        )
        return
    paths = {n: p for n, p in HOLDOUT_PATHS.items() if p.is_file()}
    skipped = [n for n in HOLDOUT_PATHS if n not in paths]
    if skipped:
        print(f"holdout comparison: skipping missing optional: {', '.join(skipped)}")
    baselines = load_from_paths(paths)
    payload = build_json(baselines, protocol_note=HOLDOUT_PROTOCOL_NOTE)
    names = list(paths)
    HOLDOUT_OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    HOLDOUT_OUT_MD.write_text(
        build_md(
            payload,
            names=names,
            labels=HOLDOUT_LABELS,
            title="# Baseline vs. twotower — matched real-holdout comparison",
            sources_note=(
                "Sources: `artifacts/{tfidf,bert_frozen,voyage_nano,voyage_large,"
                "hybrid_tfidf_voyage}_holdout/metrics.json` + "
                "`artifacts/twotower/<run_id>_holdout_eval/metrics_holdout.json`."
            ),
        )
    )
    print(f"wrote {HOLDOUT_OUT_JSON.relative_to(ROOT)}")
    print(f"wrote {HOLDOUT_OUT_MD.relative_to(ROOT)}")
    print(f"holdout comparison: {', '.join(names)}")


def main() -> None:
    baselines = load_baselines()
    payload = build_json(baselines)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    OUT_MD.write_text(build_md(payload))
    print(f"wrote {OUT_JSON.relative_to(ROOT)}")
    print(f"wrote {OUT_MD.relative_to(ROOT)}")
    print(f"baselines: {', '.join(baselines)}")

    add_twotower_holdout_run("run_001")
    add_twotower_holdout_run("arm_a_real_only")
    build_holdout_comparison()


if __name__ == "__main__":
    main()
