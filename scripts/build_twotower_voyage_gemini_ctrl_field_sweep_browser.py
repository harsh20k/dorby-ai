#!/usr/bin/env python3
"""Build a self-contained HTML heatmap for the voyage_gemini_ctrl field/query sweep.

Reads artifacts/twotower_voyage_gemini_ctrl_field_sweep_modal/real_all/sweep_results.json
(105 combos x full metric suite — 7 non-empty field subsets per side x 7 x 2
query settings, plus a query-only seeker x the same 7 candidate subsets),
scored against the fine-tuned ``voyage_gemini_ctrl_001`` LoRA checkpoint —
``top1_ctrl``'s exact recipe retrained on a bigger, newer synthetic batch,
holder of the best raw full-profile pair AUC of any fine-tune in the
project (docs/twotower-voyage-gemini-ctrl-experiment.md) — and renders an
8x7 seeker x candidate heatmap (metric-selectable: any of the tracked
pair/retrieval metrics can drive cell color) with side-by-side
query-included/excluded heatmaps, cell hover detail, and a full 105-row
data table — same visual language (tokens, table styling, interactive
controls) as scripts/build_twotower_queryonly_back_look_field_sweep_browser.py,
copied and repointed at the new checkpoint's results rather than edited in
place (isolation rule, CLAUDE.md).

Usage:
    python3 scripts/build_twotower_voyage_gemini_ctrl_field_sweep_browser.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = (
    ROOT
    / "artifacts"
    / "twotower_voyage_gemini_ctrl_field_sweep_modal"
    / "real_all"
    / "sweep_results.json"
)
DEFAULT_OUT = ROOT / "docs" / "html" / "twotower-voyage-gemini-ctrl-field-sweep-heatmap.html"

FIELDS = ("positioning", "background", "lookingFor")
SHORT = {"positioning": "Pos", "background": "Back", "lookingFor": "Look"}

# Candidate axis: 7 non-empty subsets, single fields first, then pairs, then
# all three — matches baselines/voyage_nano_field_sweep/fields.py's
# FIELD_SUBSETS order.
CANDIDATE_AXIS_ORDER = [
    ("positioning",),
    ("background",),
    ("lookingFor",),
    ("positioning", "background"),
    ("positioning", "lookingFor"),
    ("background", "lookingFor"),
    ("positioning", "background", "lookingFor"),
]
# Seeker axis: the same 7, plus the empty ("query-only") subset first —
# matches fields.py's SEEKER_SUBSETS order.
SEEKER_AXIS_ORDER = [()] + CANDIDATE_AXIS_ORDER

# All metrics the interactive controls can select, shared with the JS side.
# kind drives formatting: "auc" -> 4-decimal, "pct" -> percentage.
METRIC_LABELS: dict[str, tuple[str, str]] = {
    "pair_auc": ("Pair AUC", "auc"),
    "hard_auc": ("Hard-negative AUC", "auc"),
    "easy_auc": ("Easy-negative AUC", "auc"),
    "best_f1": ("Best F1", "auc"),
    "accuracy_at_half": ("Accuracy @ 0.5", "auc"),
    "mrr": ("MRR", "auc"),
    "recall_1": ("Recall@1", "pct"),
    "recall_5": ("Recall@5", "pct"),
    "recall_10": ("Recall@10", "pct"),
    "ndcg_1": ("NDCG@1", "auc"),
    "ndcg_5": ("NDCG@5", "auc"),
    "ndcg_10": ("NDCG@10", "auc"),
}
METRIC_KEYS = list(METRIC_LABELS.keys())


def axis_label(fields: tuple[str, ...]) -> str:
    return "+".join(SHORT[f] for f in fields) or "None"


def _row_from_result(r: dict) -> dict:
    """Full metric row for one combo — shared by grids, full_rows, and combo_lookup
    so the interactive selector and the heatmap/table always agree on values."""
    p = r["pair"]
    h = r["slices"]["neg_hardness"]
    ret = r["retrieval"]
    return {
        "seeker": axis_label(tuple(r["seeker_fields"])),
        "candidate": axis_label(tuple(r["candidate_fields"])),
        "use_query": r["use_query"],
        "pair_auc": p["roc_auc"],
        "best_f1": p["best_f1"],
        "accuracy_at_half": p["accuracy_at_0.5"],
        "hard_auc": h["hard"].get("pair_auc"),
        "easy_auc": h["easy"].get("pair_auc"),
        "mrr": ret["mrr"],
        "mean_rank": ret["mean_rank"],
        "median_rank": ret["median_rank"],
        "recall_1": ret["recall@1"],
        "recall_5": ret["recall@5"],
        "recall_10": ret["recall@10"],
        "ndcg_1": ret["ndcg@1"],
        "ndcg_5": ret["ndcg@5"],
        "ndcg_10": ret["ndcg@10"],
    }


def _combo_key(seeker_fields: tuple[str, ...], candidate_fields: tuple[str, ...], use_query: bool) -> str:
    return f"{axis_label(seeker_fields)}|{axis_label(candidate_fields)}|{'yes' if use_query else 'no'}"


def _na_cell(seeker_fields: tuple[str, ...], candidate_fields: tuple[str, ...]) -> dict:
    """Placeholder for a (seeker, candidate, query=no) cell that was never computed —
    only the query-only seeker (empty fields) + query=False combo, which is
    degenerate (fully empty seeker text) and skipped by fields.py."""
    cell = {k: None for k in METRIC_KEYS}
    cell.update(
        {
            "seeker": axis_label(seeker_fields),
            "candidate": axis_label(candidate_fields),
            "use_query": False,
            "mean_rank": None,
            "median_rank": None,
            "na": True,
        }
    )
    return cell


def build_data() -> dict:
    results = json.loads(RESULTS_PATH.read_text())

    by_key: dict[tuple[tuple[str, ...], tuple[str, ...], bool], dict] = {}
    for r in results:
        key = (tuple(r["seeker_fields"]), tuple(r["candidate_fields"]), r["use_query"])
        by_key[key] = r

    combo_lookup: dict[str, dict] = {}
    for r in results:
        key = _combo_key(tuple(r["seeker_fields"]), tuple(r["candidate_fields"]), r["use_query"])
        combo_lookup[key] = _row_from_result(r)

    grids = {}
    for use_query in (True, False):
        grid = []
        for seeker_fields in SEEKER_AXIS_ORDER:
            row = []
            for candidate_fields in CANDIDATE_AXIS_ORDER:
                r = by_key.get((seeker_fields, candidate_fields, use_query))
                row.append(_row_from_result(r) if r is not None else _na_cell(seeker_fields, candidate_fields))
            grid.append(row)
        grids["yes" if use_query else "no"] = grid

    all_auc = [r["pair"]["roc_auc"] for r in results]
    full_rows = [_row_from_result(r) for r in results]
    full_rows.sort(key=lambda r: -r["pair_auc"])

    metric_ranges = {}
    for key in METRIC_KEYS:
        values = [r[key] for r in full_rows if r[key] is not None]
        metric_ranges[key] = {"min": min(values), "max": max(values)}

    return {
        "seeker_axis_labels": [axis_label(f) for f in SEEKER_AXIS_ORDER],
        "candidate_axis_labels": [axis_label(f) for f in CANDIDATE_AXIS_ORDER],
        "fields": list(FIELDS),
        "field_short": SHORT,
        "grids": grids,
        "combo_lookup": combo_lookup,
        "metric_ranges": metric_ranges,
        "metric_options": [[key, *METRIC_LABELS[key]] for key in METRIC_KEYS],
        "min_auc": min(all_auc),
        "max_auc": max(all_auc),
        "full_rows": full_rows,
        "num_combos": len(results),
    }


# Sequential blue ramp (references/palette.md), light->dark, used for
# continuous interpolation across the observed pair-AUC range.
RAMP_LIGHT = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
# Dark-mode surface companion ramp: same hue family, values chosen to read
# against the dark chart surface (#1a1a19) — lightest step still >=3:1.
RAMP_DARK = [
    "#123a63", "#154577", "#18508a", "#1c5cab", "#256abf", "#2a78d6",
    "#3987e5", "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4", "#b7d3f6", "#cde2fb",
]

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>voyage_gemini_ctrl field/query sweep — Dorby AI</title>
<style>
  :root {
    color-scheme: light;
    --surface-1: #fcfcfb;
    --page: #f9f9f7;
    --ink: #0b0b0b;
    --ink-2: #52514e;
    --muted: #898781;
    --grid: #e1e0d9;
    --border: rgba(11,11,11,0.10);
    --panel: #ffffff;
    --good: #0ca30c;
    --good-bg: #e3f6e3;
  }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) {
      color-scheme: dark;
      --surface-1: #1a1a19;
      --page: #0d0d0d;
      --ink: #ffffff;
      --ink-2: #c3c2b7;
      --muted: #898781;
      --grid: #2c2c2a;
      --border: rgba(255,255,255,0.10);
      --panel: #161615;
      --good: #0ca30c;
      --good-bg: #103010;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --surface-1: #1a1a19;
    --page: #0d0d0d;
    --ink: #ffffff;
    --ink-2: #c3c2b7;
    --muted: #898781;
    --grid: #2c2c2a;
    --border: rgba(255,255,255,0.10);
    --panel: #161615;
    --good: #0ca30c;
    --good-bg: #103010;
  }

  * { box-sizing: border-box; }
  html, body { margin: 0; min-height: 100%; }
  body {
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    color: var(--ink);
    background: var(--page);
    line-height: 1.45;
  }
  :where(a, button, input, [tabindex]):focus-visible {
    outline: 2px solid var(--good);
    outline-offset: 2px;
    border-radius: 4px;
  }
  @media (prefers-reduced-motion: reduce) { * { transition: none !important; } }

  .wrap { width: min(1180px, calc(100% - 2rem)); margin: 0 auto; padding: 2rem 0 4rem; }

  header.hero { display: grid; gap: 0.6rem; margin-bottom: 1.6rem; }
  .brand { font-size: clamp(1.6rem, 3.6vw, 2.2rem); letter-spacing: -0.02em; margin: 0; font-weight: 700; display: flex; align-items: baseline; gap: 0.6rem; flex-wrap: wrap; }
  .n-badge {
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.02em; text-transform: uppercase;
    color: var(--good); background: var(--good-bg); border-radius: 999px; padding: 0.25rem 0.6rem;
    vertical-align: middle;
  }
  .lede { margin: 0; max-width: 62rem; color: var(--ink-2); font-size: 0.95rem; }
  .lede code { background: var(--surface-1); border: 1px solid var(--border); border-radius: 4px; padding: 0.05rem 0.35rem; font-size: 0.85em; }
  .meta { margin: 0; font-size: 0.8rem; color: var(--muted); }

  .tabs { display: flex; gap: 0.5rem; margin-top: 1rem; }
  .tab-btn {
    font-size: 0.82rem; font-weight: 600; padding: 0.45rem 0.9rem; border-radius: 8px;
    border: 1px solid var(--border); background: var(--panel); color: var(--ink-2); cursor: pointer;
  }
  .tab-btn.active { color: var(--ink); border-color: var(--ink); background: var(--surface-1); }
  .tab-btn:hover { background: var(--surface-1); }

  .explore-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem; }
  @media (max-width: 900px) { .explore-grid { grid-template-columns: 1fr; } }

  .field-group { display: flex; flex-direction: column; gap: 0.5rem; }
  .field-group-label { font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.02em; color: var(--muted); }
  .checkbox-row { display: flex; gap: 0.9rem; flex-wrap: wrap; }
  .checkbox-pill {
    display: inline-flex; align-items: center; gap: 0.4rem; font-size: 0.85rem;
    padding: 0.35rem 0.7rem; border-radius: 8px; border: 1px solid var(--border);
    background: var(--surface-1); cursor: pointer; user-select: none;
  }
  .checkbox-pill input { accent-color: var(--good); width: 0.95rem; height: 0.95rem; }
  .checkbox-pill.checked { border-color: var(--good); background: var(--good-bg); }

  .query-toggle-row { display: flex; align-items: center; gap: 0.6rem; margin-top: 0.2rem; }
  .switch { position: relative; display: inline-block; width: 2.4rem; height: 1.35rem; flex: none; }
  .switch input { opacity: 0; width: 0; height: 0; }
  .switch-track {
    position: absolute; inset: 0; background: var(--grid); border-radius: 999px; transition: background 0.15s ease;
  }
  .switch-track::before {
    content: ""; position: absolute; width: 1.05rem; height: 1.05rem; left: 0.15rem; top: 0.15rem;
    background: var(--panel); border-radius: 50%; transition: transform 0.15s ease; box-shadow: 0 1px 2px rgba(0,0,0,0.2);
  }
  .switch input:checked + .switch-track { background: var(--good); }
  .switch input:checked + .switch-track::before { transform: translateX(1.05rem); }

  .result-card { margin-top: 0.9rem; padding: 1rem; border-radius: 10px; background: var(--surface-1); border: 1px solid var(--border); }
  .result-card.is-empty { color: var(--muted); font-size: 0.85rem; text-align: center; padding: 1.6rem 1rem; }
  .result-headline { font-size: 0.9rem; font-weight: 700; margin: 0 0 0.7rem; }
  .stat-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.6rem; }
  .stat-tile { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 0.55rem 0.7rem; }
  .stat-tile .k { font-size: 0.66rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.02em; color: var(--muted); margin: 0 0 0.15rem; }
  .stat-tile .v {
    font-size: 1rem; font-weight: 700; font-family: ui-monospace, "SF Mono", "Cascadia Code", Menlo, Consolas, monospace;
    font-variant-numeric: tabular-nums;
  }

  .metric-select-row { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.8rem; }
  .metric-select {
    font-size: 0.85rem; font-weight: 600; padding: 0.4rem 0.7rem; border-radius: 8px;
    border: 1px solid var(--border); background: var(--surface-1); color: var(--ink); cursor: pointer;
  }
  .top3-list { display: flex; flex-direction: column; gap: 0.55rem; }
  .top3-row {
    display: flex; align-items: center; gap: 0.7rem; padding: 0.6rem 0.75rem; border-radius: 8px;
    background: var(--surface-1); border: 1px solid var(--border);
  }
  .top3-rank {
    flex: none; width: 1.6rem; height: 1.6rem; border-radius: 50%; background: var(--good-bg); color: var(--good);
    font-weight: 700; font-size: 0.82rem; display: flex; align-items: center; justify-content: center;
  }
  .top3-body { flex: 1; }
  .top3-combo { font-size: 0.84rem; font-weight: 600; color: var(--ink); }
  .top3-summary { font-size: 0.78rem; color: var(--muted); margin-top: 0.1rem; }
  .top3-value {
    font-family: ui-monospace, "SF Mono", "Cascadia Code", Menlo, Consolas, monospace;
    font-variant-numeric: tabular-nums; font-weight: 700; font-size: 0.92rem; flex: none;
  }

  .chart-card {
    background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
    padding: 1.4rem 1.4rem 1.2rem; margin-top: 1rem;
  }
  .chart-title { font-size: 0.95rem; font-weight: 700; margin: 0 0 0.15rem; }
  .chart-sub { font-size: 0.78rem; color: var(--muted); margin: 0 0 1rem; }

  .heatmap-pair-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.4rem; margin-top: 0.6rem; }
  @media (max-width: 900px) { .heatmap-pair-grid { grid-template-columns: 1fr; } }
  .heatmap-pane-title { font-size: 0.8rem; font-weight: 700; margin: 0 0 0.5rem; color: var(--ink-2); }
  .heatmap-scroll { overflow-x: auto; }
  .heatmap-layout { display: grid; grid-template-columns: auto 1fr; gap: 0.5rem; align-items: start; min-width: 640px; }
  .axis-label-y {
    display: flex; flex-direction: column; justify-content: center; align-items: flex-end;
    font-size: 0.72rem; font-weight: 700; color: var(--muted); writing-mode: vertical-rl;
    transform: rotate(180deg); text-align: center; padding-right: 0.3rem;
  }
  .heatmap-col-labels { display: grid; grid-auto-flow: column; margin-bottom: 0.3rem; }
  .heatmap-row { display: grid; grid-auto-flow: column; align-items: center; }
  .row-label {
    font-size: 0.74rem; font-weight: 700; color: var(--ink-2); text-align: right;
    padding-right: 0.5rem; white-space: nowrap;
  }
  .col-label {
    font-size: 0.74rem; font-weight: 700; color: var(--ink-2); text-align: center;
    padding-bottom: 0.3rem; white-space: nowrap;
  }
  .heatmap-grid { display: grid; gap: 3px; }
  .cell {
    aspect-ratio: 1.15 / 1; min-width: 5.4rem; border-radius: 6px; display: flex;
    align-items: center; justify-content: center; font-size: 0.78rem; font-weight: 700;
    font-family: ui-monospace, "SF Mono", "Cascadia Code", Menlo, Consolas, monospace;
    cursor: default; border: 2px solid transparent;
  }
  .cell:hover, .cell:focus-visible { border-color: var(--ink); outline: none; }
  .heatmap-body { display: grid; grid-template-columns: repeat(7, 1fr); gap: 3px; }

  .legend-bar { display: flex; align-items: center; gap: 0.6rem; margin-top: 0.9rem; font-size: 0.74rem; color: var(--muted); }

  .heatmap-grid-all { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.1rem; margin-top: 0.6rem; }
  @media (max-width: 980px) { .heatmap-grid-all { grid-template-columns: repeat(2, 1fr); } }
  @media (max-width: 620px) { .heatmap-grid-all { grid-template-columns: 1fr; } }
  .mini-card { border: 1px solid var(--border); border-radius: 10px; padding: 0.8rem 0.8rem 0.7rem; background: var(--panel); }
  .mini-title { font-size: 0.76rem; font-weight: 700; margin: 0 0 0.55rem; color: var(--ink); }
  .mini-layout { display: grid; grid-template-columns: 3.2rem 1fr; gap: 0.25rem; align-items: stretch; }
  .mini-col-labels { display: grid; gap: 2px; margin-left: 3.2rem; margin-bottom: 2px; }
  .mini-col-label {
    font-size: 0.52rem; font-weight: 700; color: var(--muted); text-align: center;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  .mini-row-labels { display: grid; gap: 2px; }
  .mini-row-label {
    font-size: 0.54rem; font-weight: 700; color: var(--muted); display: flex;
    align-items: center; justify-content: flex-end; padding-right: 0.2rem; white-space: nowrap;
  }
  .mini-body { display: grid; gap: 2px; }
  .mini-cell { aspect-ratio: 1 / 1; min-width: 1.5rem; border-radius: 3px; cursor: default; }
  .mini-cell:hover, .mini-cell:focus-visible { outline: 2px solid var(--ink); outline-offset: -1px; }
  .mini-legend { display: flex; align-items: center; gap: 0.4rem; margin-top: 0.5rem; font-size: 0.62rem; color: var(--muted); }
  .mini-legend-ramp { flex: 1; height: 0.4rem; border-radius: 3px; }
  .legend-ramp { flex: 1; max-width: 22rem; height: 0.7rem; border-radius: 4px; }

  .fields-tooltip {
    position: fixed; z-index: 50; pointer-events: none; opacity: 0; visibility: hidden;
    transform: translateY(4px); transition: opacity 0.1s ease, transform 0.1s ease;
    background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.18); padding: 0.65rem 0.8rem; max-width: min(24rem, 90vw);
  }
  .fields-tooltip.is-visible { opacity: 1; visibility: visible; transform: translateY(0); }
  .fields-tooltip .tt-name { font-size: 0.8rem; font-weight: 700; margin: 0 0 0.4rem; color: var(--ink); }
  .fields-tooltip table { border-collapse: collapse; width: 100%; }
  .fields-tooltip th {
    text-align: left; font-size: 0.64rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.02em; color: var(--muted); padding: 0 0.6rem 0.25rem 0; white-space: nowrap;
  }
  .fields-tooltip td {
    text-align: left; font-size: 0.76rem; color: var(--ink); padding: 0 0.6rem 0.15rem 0;
    font-family: ui-monospace, "SF Mono", "Cascadia Code", Menlo, Consolas, monospace;
    font-variant-numeric: tabular-nums; white-space: nowrap;
  }

  .section-title { margin: 1.7rem 0 0.2rem; font-size: 0.95rem; font-weight: 700; }
  .section-sub { margin: 0 0 0.4rem; font-size: 0.78rem; color: var(--muted); }

  .findings { display: grid; gap: 0.7rem; margin-top: 0.6rem; }
  .finding {
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
    padding: 0.9rem 1.1rem; font-size: 0.87rem; color: var(--ink-2);
  }
  .finding b { color: var(--ink); }
  .finding code { background: var(--surface-1); border: 1px solid var(--border); border-radius: 4px; padding: 0.02rem 0.3rem; font-size: 0.85em; }

  .table-scroll {
    margin-top: 0.8rem; overflow-x: auto; border: 1px solid var(--border);
    border-radius: 10px; background: var(--panel); max-height: 32rem; overflow-y: auto;
  }
  table.data-table { border-collapse: collapse; width: 100%; font-size: 0.84rem; }
  table.data-table thead th {
    position: sticky; top: 0; background: var(--surface-1); text-align: right;
    padding: 0.6rem 0.8rem; font-weight: 700; white-space: nowrap;
    border-bottom: 1px solid var(--grid);
  }
  table.data-table thead th:first-child, table.data-table thead th:nth-child(2), table.data-table thead th:nth-child(3) { text-align: left; }
  table.data-table tbody td, table.data-table tbody th {
    padding: 0.45rem 0.8rem; border-bottom: 1px solid var(--grid); white-space: nowrap;
  }
  table.data-table tbody th { text-align: left; font-weight: 600; color: var(--ink-2); }
  table.data-table tbody td {
    text-align: right; color: var(--ink);
    font-family: ui-monospace, "SF Mono", "Cascadia Code", Menlo, Consolas, monospace;
    font-variant-numeric: tabular-nums; font-size: 0.82rem;
  }
  table.data-table tbody tr:hover td, table.data-table tbody tr:hover th { background: color-mix(in srgb, var(--ink) 4%, transparent); }
  table.data-table tbody tr:nth-child(-n+1) td, table.data-table tbody tr:nth-child(-n+1) th { background: var(--good-bg); }

  @media (max-width: 640px) {
    .wrap { width: min(100%, calc(100% - 1.25rem)); padding-top: 1.25rem; }
  }
</style>
</head>
<body>
  <div class="fields-tooltip" id="fields-tooltip" role="tooltip"></div>
  <div class="wrap">
    <header class="hero">
      <h1 class="brand">voyage_gemini_ctrl field/query sweep <span class="n-badge" id="combo-count-badge">— combos · n = 200 (all)</span></h1>
      <p class="lede">
        Which combination of seeker fields, candidate fields, and searchQuery
        inclusion works best for <strong>voyage_gemini_ctrl_001</strong> —
        <code>top1_ctrl</code>'s exact recipe retrained on a bigger, newer
        synthetic batch, holder of the best raw full-profile pair AUC of any
        fine-tune in the project? The exact same 105-combo grid re-run
        against this checkpoint instead: every non-empty subset of
        <code>positioning</code> / <code>background</code> / <code>lookingFor</code>
        (7 per side) x with/without the search query, plus a "query-only"
        seeker (no profile fields at all) x the same 7 candidate subsets,
        each scored with the project's full metric suite on all 200 real
        pairs. <strong>This grid's best combo (0.6855 pair AUC) is a new
        project-wide record — the first embedding-based model to beat the
        LLM judge's own best pair AUC (0.6451).</strong> See
        <code>docs/twotower-voyage-gemini-ctrl-field-sweep-experiment.md</code>
        (and <code>docs/twotower-queryonly-back-look-field-sweep-experiment.md</code>
        for the previous-best-checkpoint comparison).
      </p>
      <p class="meta" id="meta-line"></p>
    </header>

    <main>
      <div class="chart-card">
        <p class="chart-title">Explore a combination</p>
        <p class="chart-sub">Pick seeker fields, candidate fields, and whether the search query is included — see that exact combo's results below.</p>
        <div class="explore-grid">
          <div>
            <div class="field-group">
              <span class="field-group-label">Seeker fields</span>
              <div class="checkbox-row" id="seeker-checkboxes"></div>
            </div>
            <div class="field-group" style="margin-top: 0.9rem;">
              <span class="field-group-label">Candidate fields</span>
              <div class="checkbox-row" id="candidate-checkboxes"></div>
            </div>
            <div class="query-toggle-row">
              <label class="switch">
                <input type="checkbox" id="query-toggle-input" checked />
                <span class="switch-track"></span>
              </label>
              <span class="field-group-label" style="text-transform:none; font-weight:600; font-size:0.85rem; color:var(--ink-2);">Include search query</span>
            </div>
          </div>
          <div id="result-card-wrap"></div>
        </div>
      </div>

      <div class="chart-card">
        <p class="chart-title">Top 3 combos by metric</p>
        <p class="chart-sub" id="top3-sub">Rank all combos by any single metric and see the top three, independent of the selection above.</p>
        <div class="metric-select-row">
          <span class="field-group-label" style="text-transform:none; font-weight:600; font-size:0.85rem; color:var(--ink-2);">Rank by</span>
          <select class="metric-select" id="metric-select"></select>
        </div>
        <div class="top3-list" id="top3-list"></div>
      </div>

      <div class="chart-card">
        <p class="chart-title" id="heatmap-title">Pair ROC-AUC heatmaps — seeker fields (rows) x candidate fields (columns)</p>
        <p class="chart-sub" id="heatmap-sub">Darker = higher pair AUC. Hover any cell for the full metric breakdown (hard/easy-neg AUC, best-F1, MRR, Recall@1/5/10, NDCG@1/5/10). Query included and excluded shown side by side.</p>
        <div class="metric-select-row">
          <span class="field-group-label" style="text-transform:none; font-weight:600; font-size:0.85rem; color:var(--ink-2);">Color / value by</span>
          <select class="metric-select" id="heatmap-metric-select"></select>
        </div>
        <div class="heatmap-pair-grid">
          <div class="heatmap-pane">
            <p class="heatmap-pane-title">Query included</p>
            <div class="heatmap-scroll">
              <div class="heatmap-layout">
                <div class="axis-label-y">Seeker fields</div>
                <div>
                  <div class="heatmap-col-labels" id="col-labels-yes" style="display:grid; grid-template-columns: repeat(7, 1fr); margin-left: 6.5rem;"></div>
                  <div style="display:grid; grid-template-columns: 6.5rem 1fr; gap: 0.3rem; align-items: stretch;" id="heatmap-rows-wrap-yes">
                    <div></div>
                    <div class="heatmap-body" id="heatmap-body-yes"></div>
                  </div>
                </div>
              </div>
            </div>
          </div>
          <div class="heatmap-pane">
            <p class="heatmap-pane-title">Query excluded</p>
            <div class="heatmap-scroll">
              <div class="heatmap-layout">
                <div class="axis-label-y">Seeker fields</div>
                <div>
                  <div class="heatmap-col-labels" id="col-labels-no" style="display:grid; grid-template-columns: repeat(7, 1fr); margin-left: 6.5rem;"></div>
                  <div style="display:grid; grid-template-columns: 6.5rem 1fr; gap: 0.3rem; align-items: stretch;" id="heatmap-rows-wrap-no">
                    <div></div>
                    <div class="heatmap-body" id="heatmap-body-no"></div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
        <div class="legend-bar">
          <span id="legend-min">—</span>
          <div class="legend-ramp" id="legend-ramp"></div>
          <span id="legend-max">—</span>
          <span id="legend-metric-label" style="margin-left:0.4rem;">pair AUC</span>
        </div>
      </div>

      <div class="chart-card">
        <p class="chart-title">All metrics at a glance</p>
        <p class="chart-sub">Every tracked metric as its own heatmap, same seeker (rows) x candidate (columns) layout as above, each with its own color scale. Hover any cell for the full breakdown.</p>
        <div class="tabs" id="query-tabs">
          <button class="tab-btn active" data-query="yes">Query included</button>
          <button class="tab-btn" data-query="no">Query excluded</button>
        </div>
        <div class="heatmap-grid-all" id="heatmap-grid-all"></div>
      </div>

      <p class="section-title">Findings</p>
      <div class="findings" id="findings"></div>

      <p class="section-title">All combos <span class="n-badge" id="table-count-badge">ranked by pair AUC</span></p>
      <p class="section-sub">Top row highlighted. Scroll for the full grid.</p>
      <div class="table-scroll">
        <table class="data-table" id="full-table"></table>
      </div>
    </main>
  </div>

  <script id="data" type="application/json">__EMBEDDED_JSON__</script>
  <script>
    const DATA = JSON.parse(document.getElementById("data").textContent);
    const RAMP_LIGHT = __RAMP_LIGHT__;
    const RAMP_DARK = __RAMP_DARK__;

    document.getElementById("meta-line").textContent =
      `${DATA.num_combos} combinations x full metric suite (pair AUC, best-F1, accuracy@0.5, hard/easy-neg AUC, MRR, mean/median rank, Recall@1/5/10, NDCG@1/5/10) — voyage_gemini_ctrl_001 (fine-tuned), all 200 real pairs.`;

    function fmt(x, digits = 4) {
      return x === null || x === undefined || Number.isNaN(x) ? "—" : x.toFixed(digits);
    }
    function pct(x, digits = 1) {
      return x === null || x === undefined ? "—" : (x * 100).toFixed(digits) + "%";
    }

    function isDark() {
      const stamp = document.documentElement.getAttribute("data-theme");
      if (stamp === "dark") return true;
      if (stamp === "light") return false;
      return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
    }

    function hexToRgb(hex) {
      const n = parseInt(hex.slice(1), 16);
      return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
    }
    function rgbToHex([r, g, b]) {
      return "#" + [r, g, b].map((v) => Math.round(v).toString(16).padStart(2, "0")).join("");
    }
    function lerpRamp(ramp, t) {
      t = Math.max(0, Math.min(1, t));
      const scaled = t * (ramp.length - 1);
      const i0 = Math.floor(scaled);
      const i1 = Math.min(ramp.length - 1, i0 + 1);
      const f = scaled - i0;
      const a = hexToRgb(ramp[i0]);
      const b = hexToRgb(ramp[i1]);
      return rgbToHex(a.map((v, idx) => v + (b[idx] - v) * f));
    }
    function textColorFor(t) {
      // t close to 1 (dark cell) -> light text; t close to 0 (light cell) -> dark text
      return t > 0.55 ? "#ffffff" : "#0b0b0b";
    }

    const METRIC_OPTIONS = DATA.metric_options; // [key, label, kind][]
    function metricInfo(key) {
      return METRIC_OPTIONS.find(([k]) => k === key) || METRIC_OPTIONS[0];
    }
    function fmtByKind(v, kind) {
      if (v === null || v === undefined) return "—";
      if (kind === "pct") return pct(v);
      return fmt(v);
    }

    let currentMetric = "pair_auc";

    function buildLegend() {
      const ramp = isDark() ? RAMP_DARK : RAMP_LIGHT;
      const el = document.getElementById("legend-ramp");
      el.style.background = `linear-gradient(to right, ${ramp.join(",")})`;
      const range = DATA.metric_ranges[currentMetric];
      const [, label] = metricInfo(currentMetric);
      document.getElementById("legend-min").textContent = fmt(range.min);
      document.getElementById("legend-max").textContent = fmt(range.max);
      document.getElementById("legend-metric-label").textContent = label;
    }

    // ---- shared hover tooltip ----
    const tooltipEl = document.getElementById("fields-tooltip");
    function showTooltip(cellData, evt) {
      if (cellData.na) {
        tooltipEl.innerHTML = `
          <p class="tt-name">Seeker: ${cellData.seeker} &middot; Candidate: ${cellData.candidate}</p>
          <p style="font-size:0.78rem; color:var(--muted); margin:0;">Not applicable — a query-only seeker with the query excluded has no seeker text at all.</p>
        `;
      } else {
        tooltipEl.innerHTML = `
          <p class="tt-name">Seeker: ${cellData.seeker} &middot; Candidate: ${cellData.candidate}</p>
          <table>
            <thead><tr><th>Pair AUC</th><th>Hard-neg</th><th>Easy-neg</th><th>Best-F1</th><th>Acc@0.5</th></tr></thead>
            <tbody><tr>
              <td>${fmt(cellData.pair_auc)}</td>
              <td>${fmt(cellData.hard_auc)}</td>
              <td>${fmt(cellData.easy_auc)}</td>
              <td>${fmt(cellData.best_f1)}</td>
              <td>${fmt(cellData.accuracy_at_half)}</td>
            </tr></tbody>
          </table>
          <table style="margin-top:0.4rem;">
            <thead><tr><th>MRR</th><th>Mean rank</th><th>R@1</th><th>R@5</th><th>R@10</th></tr></thead>
            <tbody><tr>
              <td>${fmt(cellData.mrr)}</td>
              <td>${fmt(cellData.mean_rank, 2)}</td>
              <td>${pct(cellData.recall_1)}</td>
              <td>${pct(cellData.recall_5)}</td>
              <td>${pct(cellData.recall_10)}</td>
            </tr></tbody>
          </table>
        `;
      }
      tooltipEl.classList.add("is-visible");
      positionTooltip(evt);
    }
    function positionTooltip(evt) {
      const pad = 14;
      const rect = tooltipEl.getBoundingClientRect();
      let x = evt.clientX + pad;
      let y = evt.clientY + pad;
      if (x + rect.width > window.innerWidth - pad) x = evt.clientX - rect.width - pad;
      if (y + rect.height > window.innerHeight - pad) y = evt.clientY - rect.height - pad;
      tooltipEl.style.left = `${Math.max(pad, x)}px`;
      tooltipEl.style.top = `${Math.max(pad, y)}px`;
    }
    function hideTooltip() { tooltipEl.classList.remove("is-visible"); }

    // ---- heatmap (query included / excluded panes, side by side) ----
    function renderHeatmapPane(queryKey, metricKey) {
      const suffix = queryKey === "yes" ? "yes" : "no";
      const grid = DATA.grids[queryKey];
      const ramp = isDark() ? RAMP_DARK : RAMP_LIGHT;
      const range = DATA.metric_ranges[metricKey];
      const min = range.min, max = range.max;
      const [, , metricKind] = metricInfo(metricKey);

      const colLabels = document.getElementById(`col-labels-${suffix}`);
      colLabels.style.gridTemplateColumns = `repeat(${DATA.candidate_axis_labels.length}, 1fr)`;
      colLabels.innerHTML = DATA.candidate_axis_labels.map((l) => `<div class="col-label">${l}</div>`).join("");

      const rowsWrap = document.getElementById(`heatmap-rows-wrap-${suffix}`);
      rowsWrap.innerHTML = "";
      const rowLabelsCol = document.createElement("div");
      rowLabelsCol.style.display = "grid";
      rowLabelsCol.style.gridTemplateRows = `repeat(${DATA.seeker_axis_labels.length}, 1fr)`;
      rowLabelsCol.style.gap = "3px";
      DATA.seeker_axis_labels.forEach((l) => {
        const d = document.createElement("div");
        d.className = "row-label";
        d.textContent = l;
        d.style.display = "flex";
        d.style.alignItems = "center";
        d.style.justifyContent = "flex-end";
        rowLabelsCol.appendChild(d);
      });

      const body = document.createElement("div");
      body.className = "heatmap-body";
      body.style.gridTemplateColumns = `repeat(${DATA.candidate_axis_labels.length}, 1fr)`;
      grid.forEach((row) => {
        row.forEach((cellData) => {
          const cell = document.createElement("div");
          cell.className = "cell";
          cell.tabIndex = 0;
          const v = cellData[metricKey];
          if (cellData.na || v === null || v === undefined) {
            cell.style.background = "var(--grid)";
            cell.style.color = "var(--muted)";
            cell.textContent = "—";
          } else {
            const t = (v - min) / (max - min || 1);
            cell.style.background = lerpRamp(ramp, t);
            cell.style.color = textColorFor(t);
            cell.textContent = metricKind === "pct" ? pct(v, 1) : v.toFixed(3);
          }
          cell.addEventListener("mouseenter", (evt) => showTooltip(cellData, evt));
          cell.addEventListener("mousemove", positionTooltip);
          cell.addEventListener("mouseleave", hideTooltip);
          cell.addEventListener("focus", (evt) => showTooltip(cellData, evt));
          cell.addEventListener("blur", hideTooltip);
          body.appendChild(cell);
        });
      });

      rowsWrap.appendChild(rowLabelsCol);
      rowsWrap.appendChild(body);
    }

    function renderBothHeatmaps(metricKey) {
      currentMetric = metricKey;
      const [, metricLabel] = metricInfo(metricKey);
      document.getElementById("heatmap-title").textContent =
        `${metricLabel} heatmaps — seeker fields (rows) x candidate fields (columns)`;
      document.getElementById("heatmap-sub").textContent =
        `Darker = higher ${metricLabel.toLowerCase()}. Hover any cell for the full metric breakdown. Gray cells are not applicable (query-only seeker with the query excluded). Query included and excluded shown side by side.`;
      renderHeatmapPane("yes", metricKey);
      renderHeatmapPane("no", metricKey);
      buildLegend();
    }

    // ---- all-metrics mini-heatmap grid (3 per row) ----
    function renderMiniHeatmap(card, queryKey, metricKey) {
      const grid = DATA.grids[queryKey];
      const ramp = isDark() ? RAMP_DARK : RAMP_LIGHT;
      const range = DATA.metric_ranges[metricKey];
      const min = range.min, max = range.max;
      const [, metricLabel, metricKind] = metricInfo(metricKey);

      card.innerHTML = `
        <p class="mini-title">${metricLabel}</p>
        <div class="mini-layout">
          <div></div>
          <div class="mini-col-labels" id="mini-col-labels"></div>
          <div class="mini-row-labels" id="mini-row-labels"></div>
          <div class="mini-body" id="mini-body"></div>
        </div>
        <div class="mini-legend">
          <span>${fmt(min)}</span>
          <div class="mini-legend-ramp" id="mini-legend-ramp"></div>
          <span>${fmt(max)}</span>
        </div>
      `;

      const colLabels = card.querySelector("#mini-col-labels");
      colLabels.style.gridTemplateColumns = `repeat(${DATA.candidate_axis_labels.length}, 1fr)`;
      colLabels.innerHTML = DATA.candidate_axis_labels
        .map((l) => `<div class="mini-col-label" title="${l}">${l}</div>`)
        .join("");

      const rowLabels = card.querySelector("#mini-row-labels");
      rowLabels.style.gridTemplateRows = `repeat(${DATA.seeker_axis_labels.length}, 1fr)`;
      rowLabels.innerHTML = DATA.seeker_axis_labels
        .map((l) => `<div class="mini-row-label" title="${l}">${l}</div>`)
        .join("");

      const body = card.querySelector("#mini-body");
      body.style.gridTemplateColumns = `repeat(${DATA.candidate_axis_labels.length}, 1fr)`;
      body.innerHTML = "";
      grid.forEach((row) => {
        row.forEach((cellData) => {
          const cell = document.createElement("div");
          cell.className = "mini-cell";
          cell.tabIndex = 0;
          const v = cellData[metricKey];
          if (cellData.na || v === null || v === undefined) {
            cell.style.background = "var(--grid)";
          } else {
            const t = (v - min) / (max - min || 1);
            cell.style.background = lerpRamp(ramp, t);
          }
          cell.addEventListener("mouseenter", (evt) => showTooltip(cellData, evt));
          cell.addEventListener("mousemove", positionTooltip);
          cell.addEventListener("mouseleave", hideTooltip);
          cell.addEventListener("focus", (evt) => showTooltip(cellData, evt));
          cell.addEventListener("blur", hideTooltip);
          body.appendChild(cell);
        });
      });

      card.querySelector("#mini-legend-ramp").style.background =
        `linear-gradient(to right, ${ramp.join(",")})`;
    }

    function renderAllMiniHeatmaps(queryKey) {
      const container = document.getElementById("heatmap-grid-all");
      container.innerHTML = "";
      for (const [metricKey] of METRIC_OPTIONS) {
        const card = document.createElement("div");
        card.className = "mini-card";
        container.appendChild(card);
        renderMiniHeatmap(card, queryKey, metricKey);
      }
    }

    // ---- explore-a-combination panel ----
    const FIELD_LABELS = { positioning: "Positioning", background: "Background", lookingFor: "LookingFor" };

    function renderCheckboxGroup(containerId, groupName, defaultChecked) {
      const el = document.getElementById(containerId);
      el.innerHTML = DATA.fields
        .map(
          (f) => `
        <label class="checkbox-pill ${defaultChecked.includes(f) ? "checked" : ""}">
          <input type="checkbox" data-group="${groupName}" data-field="${f}" ${defaultChecked.includes(f) ? "checked" : ""} />
          ${FIELD_LABELS[f]}
        </label>`
        )
        .join("");
    }

    function selectedFields(groupName) {
      return DATA.fields.filter((f) => {
        const cb = document.querySelector(`input[data-group="${groupName}"][data-field="${f}"]`);
        return cb && cb.checked;
      });
    }

    function fieldsToLabel(fields) {
      return fields.map((f) => DATA.field_short[f]).join("+") || "None";
    }

    const RESULT_STATS = [
      ["Pair AUC", "pair_auc", "auc"],
      ["Hard-neg AUC", "hard_auc", "auc"],
      ["Easy-neg AUC", "easy_auc", "auc"],
      ["Best F1", "best_f1", "auc"],
      ["Accuracy@0.5", "accuracy_at_half", "auc"],
      ["MRR", "mrr", "auc"],
      ["Mean rank", "mean_rank", "rank"],
      ["Median rank", "median_rank", "rank"],
      ["Recall@1", "recall_1", "pct"],
      ["Recall@5", "recall_5", "pct"],
      ["Recall@10", "recall_10", "pct"],
      ["NDCG@10", "ndcg_10", "auc"],
    ];

    function fmtStat(v, kind) {
      if (v === null || v === undefined) return "—";
      if (kind === "pct") return pct(v);
      if (kind === "rank") return v.toFixed(2);
      return fmt(v);
    }

    function renderResultCard() {
      const seekerFields = selectedFields("seeker");
      const candidateFields = selectedFields("candidate");
      const useQuery = document.getElementById("query-toggle-input").checked;
      const wrap = document.getElementById("result-card-wrap");

      if (candidateFields.length === 0) {
        wrap.innerHTML = `<div class="result-card is-empty">Select at least one candidate field. (Zero seeker fields is allowed — that's the "query-only" seeker.)</div>`;
        return;
      }

      const key = `${fieldsToLabel(seekerFields)}|${fieldsToLabel(candidateFields)}|${useQuery ? "yes" : "no"}`;
      const row = DATA.combo_lookup[key];
      if (!row) {
        const reason =
          seekerFields.length === 0 && !useQuery
            ? "A query-only seeker (no fields) with the query excluded has no text at all — not a valid combination."
            : "No data for this combination.";
        wrap.innerHTML = `<div class="result-card is-empty">${reason}</div>`;
        return;
      }

      wrap.innerHTML = `
        <div class="result-card">
          <p class="result-headline">Seeker: ${fieldsToLabel(seekerFields)} &middot; Candidate: ${fieldsToLabel(candidateFields)} &middot; Query: ${useQuery ? "yes" : "no"}</p>
          <div class="stat-grid">
            ${RESULT_STATS.map(
              ([label, k, kind]) => `
              <div class="stat-tile">
                <p class="k">${label}</p>
                <p class="v">${fmtStat(row[k], kind)}</p>
              </div>`
            ).join("")}
          </div>
        </div>
      `;
    }

    renderCheckboxGroup("seeker-checkboxes", "seeker", ["positioning"]);
    renderCheckboxGroup("candidate-checkboxes", "candidate", ["lookingFor"]);
    renderResultCard();

    document.getElementById("seeker-checkboxes").addEventListener("change", (e) => {
      e.target.closest(".checkbox-pill")?.classList.toggle("checked", e.target.checked);
      renderResultCard();
    });
    document.getElementById("candidate-checkboxes").addEventListener("change", (e) => {
      e.target.closest(".checkbox-pill")?.classList.toggle("checked", e.target.checked);
      renderResultCard();
    });
    document.getElementById("query-toggle-input").addEventListener("change", renderResultCard);

    // ---- static count badges ----
    document.getElementById("combo-count-badge").textContent = `${DATA.num_combos} combos · n = 200 (all)`;
    document.getElementById("table-count-badge").textContent = `ranked by pair AUC · all ${DATA.num_combos} combos`;
    document.getElementById("top3-sub").textContent =
      `Rank all ${DATA.num_combos} combos by any single metric and see the top three, independent of the selection above.`;

    // ---- top-3-by-metric panel ----
    function renderTop3(metricKey) {
      const [, metricLabel, kind] = metricInfo(metricKey);
      const ranked = DATA.full_rows
        .filter((r) => r[metricKey] !== null && r[metricKey] !== undefined)
        .slice()
        .sort((a, b) => b[metricKey] - a[metricKey])
        .slice(0, 3);

      const listEl = document.getElementById("top3-list");
      listEl.innerHTML = ranked
        .map(
          (r, i) => `
        <div class="top3-row">
          <div class="top3-rank">${i + 1}</div>
          <div class="top3-body">
            <div class="top3-combo">Seeker: ${r.seeker} &middot; Candidate: ${r.candidate} &middot; Query: ${r.use_query ? "yes" : "no"}</div>
            <div class="top3-summary">Best ${metricLabel.toLowerCase()} among the ${DATA.num_combos} combos tested, pair AUC ${fmt(r.pair_auc)}, hard-neg AUC ${fmt(r.hard_auc)}.</div>
          </div>
          <div class="top3-value">${fmtStat(r[metricKey], kind)}</div>
        </div>
      `
        )
        .join("");
    }

    const metricSelectEl = document.getElementById("metric-select");
    metricSelectEl.innerHTML = METRIC_OPTIONS.map(([k, label]) => `<option value="${k}">${label}</option>`).join("");
    metricSelectEl.value = "pair_auc";
    renderTop3("pair_auc");
    metricSelectEl.addEventListener("change", (e) => renderTop3(e.target.value));

    // ---- heatmap controls: query tabs + metric select ----
    const heatmapMetricSelectEl = document.getElementById("heatmap-metric-select");
    heatmapMetricSelectEl.innerHTML = METRIC_OPTIONS.map(([k, label]) => `<option value="${k}">${label}</option>`).join("");
    heatmapMetricSelectEl.value = "pair_auc";

    function currentQueryTab() {
      return document.querySelector("#query-tabs .tab-btn.active")?.dataset.query || "yes";
    }

    renderBothHeatmaps(heatmapMetricSelectEl.value);
    renderAllMiniHeatmaps(currentQueryTab());

    document.getElementById("query-tabs").addEventListener("click", (e) => {
      const btn = e.target.closest(".tab-btn");
      if (!btn) return;
      document.querySelectorAll("#query-tabs .tab-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      renderAllMiniHeatmaps(btn.dataset.query);
    });
    heatmapMetricSelectEl.addEventListener("change", (e) => {
      renderBothHeatmaps(e.target.value);
    });

    // ---- findings ----
    const findingsEl = document.getElementById("findings");
    const findings = [
      `<b>New project-wide record — the first embedding-based model to beat the LLM judge's own best pair AUC.</b> The grid's best combo
       (seeker=<code>background</code>, candidate=<code>lookingFor</code>, query included) reaches <b>0.6855 pair AUC</b>, beating
       <code>top1_ctrl_field_sweep</code>'s grid best (0.6395, +0.046), <code>queryonly_back_look_field_sweep</code>'s grid best (0.6349, +0.051),
       the frozen Voyage-4-nano sweep's best (0.6110, +0.075), this checkpoint's own canonical full-profile baseline (0.6081, +0.077) — and the
       LLM judge's own best pair AUC (0.6451, <code>docs/llm-judge-focused-prompt-experiment.md</code>) by +0.040. Every prior field/query sweep
       in this project landed within or below the judge's number; this is the first to clear it.`,
      `<b>The hard/easy-neg gap on the best combo is the tightest of any top row in the project.</b> 0.6732 hard vs 0.7084 easy (grid-own
       hardness split) — both high in absolute terms, and a 0.035 gap far smaller than the typical embedding-baseline pattern (TF-IDF: 0.75 easy
       vs 0.50 hard). Not the full inversion <code>queryonly_back_look_001</code>'s training combo showed, but closer to it than any other
       fine-tune's grid-best row.`,
      `<b>A query-only seeker (zero profile fields) alone already beats the LLM judge's best.</b> 0.6716 pair AUC with just
       candidate=<code>lookingFor</code> and no seeker profile text at all — the search query alone, through this checkpoint's encoder,
       out-predicts every LLM-judge prompt variant tested in this project on pair AUC.`,
      `<b>Standard hardness-methodology caveat, carried from every prior sweep in this series.</b> This grid classifies hard/easy using each
       combo's own trimmed text, not the canonical full-profile-pinned split, so hard/easy numbers here are relative-within-the-grid only — not
       directly comparable to the project's canonical hard-neg-AUC record (0.6564, from <code>queryonly_back_look_001</code>'s pinned-split eval).`,
      `<b>Search query still matters everywhere in the grid</b> — mean pair AUC with the query included (0.6351, n=56) beats without it
       (0.5895, n=49); every one of the 5 worst combos in the grid excludes the query.`,
      `<b>Seeker field-count is monotonically worse, like <code>top1_ctrl</code>'s pattern, not <code>queryonly_back_look</code>'s plateau.</b>
       Mean pair AUC by seeker size: 0 fields (query-only) 0.6317 &rarr; 1 field 0.6144 &rarr; 2 fields 0.6141 &rarr; 3 fields (full) 0.6024.
       Candidate side is the first sweep in this project where mean AUC peaks at 1 field, not 2 (1-field 0.6168, 2-field 0.6139, 3-field 0.6047)
       — though the single best candidate field either way is still <code>lookingFor</code> alone.`,
      `<b>Caveat: 105-way search on one 200-pair population, no held-out check — and the training data itself is an unreviewed, measurably
       leakier pilot batch.</b> The gap between the best (0.6855) and 10th-best (0.6636) combo is real but not huge. This sweep answers "given
       this checkpoint, does field selection help," not "is this checkpoint's training data trustworthy" — see
       <code>docs/twotower-voyage-gemini-ctrl-experiment.md</code>'s own leakage caveats before treating 0.6855 as a validated production number.`,
    ];
    for (const f of findings) {
      const div = document.createElement("div");
      div.className = "finding";
      div.innerHTML = f;
      findingsEl.appendChild(div);
    }

    // ---- full table ----
    const cols = [
      ["Seeker", (r) => r.seeker, "left"],
      ["Candidate", (r) => r.candidate, "left"],
      ["Query", (r) => (r.use_query ? "yes" : "no"), "left"],
      ["Pair AUC", (r) => fmt(r.pair_auc)],
      ["Hard-neg", (r) => fmt(r.hard_auc)],
      ["Easy-neg", (r) => fmt(r.easy_auc)],
      ["Best F1", (r) => fmt(r.best_f1)],
      ["Acc@0.5", (r) => fmt(r.accuracy_at_half)],
      ["MRR", (r) => fmt(r.mrr)],
      ["Mean rank", (r) => fmt(r.mean_rank, 2)],
      ["R@1", (r) => pct(r.recall_1)],
      ["R@5", (r) => pct(r.recall_5)],
      ["R@10", (r) => pct(r.recall_10)],
    ];
    const tableEl = document.getElementById("full-table");
    tableEl.innerHTML = `
      <thead><tr>${cols.map(([label], i) => `<th style="${i < 3 ? 'text-align:left' : ''}">${label}</th>`).join("")}</tr></thead>
      <tbody>
        ${DATA.full_rows.map((r) => `
          <tr>
            ${cols.map(([label, fn], i) => i < 3 ? `<th>${fn(r)}</th>` : `<td>${fn(r)}</td>`).join("")}
          </tr>
        `).join("")}
      </tbody>
    `;
  </script>
</body>
</html>
"""


def main() -> int:
    data = build_data()
    html = HTML_TEMPLATE.replace("__EMBEDDED_JSON__", json.dumps(data))
    html = html.replace("__RAMP_LIGHT__", json.dumps(RAMP_LIGHT))
    html = html.replace("__RAMP_DARK__", json.dumps(RAMP_DARK))
    DEFAULT_OUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUT.write_text(html)
    print(f"wrote {DEFAULT_OUT} ({data['num_combos']} combos)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
