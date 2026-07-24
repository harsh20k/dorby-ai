#!/usr/bin/env python3
"""Build a self-contained HTML browser for docs/baseline-results-holdout.json.

Lets you toggle which models are shown, switch between Pair / Retrieval /
Neg-hardness / Intent metric views, and highlights the best value per row —
a quicker way to scan the matched-holdout comparison than the raw markdown
table in docs/baseline-results-holdout.md.

Usage:
    python3 scripts/build_holdout_browser.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = ROOT / "docs" / "baseline-results-holdout.json"
DEFAULT_OUT = ROOT / "docs" / "baseline-results-holdout-browser.html"

# Fixed categorical order (validated palette, slots 1-7) — assigned to
# models in the order they appear in the source JSON, never re-cycled.
SERIES_COLORS_LIGHT = [
    "#2a78d6", "#008300", "#e87ba4", "#eda100",
    "#1baf7a", "#eb6834", "#4a3aa7", "#e34948",
]
SERIES_COLORS_DARK = [
    "#3987e5", "#008300", "#d55181", "#c98500",
    "#199e70", "#d95926", "#9085e9", "#e66767",
]

LABELS = {
    "tfidf": "TF-IDF (lexical)",
    "bert_frozen": "Frozen BERT",
    "voyage_nano": "Voyage-4-nano",
    "voyage_large": "Voyage-4-large (prod)",
    "hybrid_tfidf_voyage": "Hybrid TF-IDF+nano",
    "twotower_run_001": "twotower run_001",
    "twotower_arm_a_real_only": "twotower arm_a_real_only",
}

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Holdout comparison browser — Dorby AI</title>
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
    --good: #0ca30c;
    --good-bg: #e3f6e3;
    --panel: #ffffff;
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
      --good: #0ca30c;
      --good-bg: #103010;
      --panel: #161615;
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
    --good: #0ca30c;
    --good-bg: #103010;
    --panel: #161615;
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
  @media (prefers-reduced-motion: reduce) {
    * { transition: none !important; }
  }

  .wrap { width: min(1280px, calc(100% - 2rem)); margin: 0 auto; padding: 2rem 0 4rem; }

  header.hero { display: grid; gap: 0.6rem; margin-bottom: 1.5rem; }
  .brand { font-size: clamp(1.6rem, 3.6vw, 2.2rem); letter-spacing: -0.02em; margin: 0; font-weight: 700; }
  .lede { margin: 0; max-width: 60rem; color: var(--ink-2); font-size: 0.95rem; }
  .lede code { background: var(--surface-1); border: 1px solid var(--border); border-radius: 4px; padding: 0.05rem 0.35rem; font-size: 0.85em; }

  .toolbar { display: flex; flex-wrap: wrap; gap: 0.9rem; align-items: flex-start; margin-top: 1.1rem; }

  .model-toggles { display: flex; flex-wrap: wrap; gap: 0.5rem; flex: 1 1 480px; }
  .model-chip {
    display: inline-flex; align-items: center; gap: 0.45rem;
    border: 1px solid var(--border); background: var(--surface-1);
    border-radius: 999px; padding: 0.35rem 0.75rem 0.35rem 0.55rem;
    font-size: 0.82rem; cursor: pointer; user-select: none;
    transition: opacity 120ms ease;
  }
  .model-chip.off { opacity: 0.4; }
  .model-chip .dot { width: 0.65rem; height: 0.65rem; border-radius: 50%; flex: none; }
  .model-chip input {
    position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px;
    overflow: hidden; clip: rect(0,0,0,0); white-space: nowrap; border: 0;
  }
  .model-chip:has(input:focus-visible) {
    outline: 2px solid var(--good); outline-offset: 2px;
  }

  .search {
    display: flex; align-items: center; gap: 0.5rem;
    border: 1px solid var(--border); background: var(--surface-1);
    border-radius: 999px; padding: 0.45rem 0.9rem; flex: 0 0 260px;
  }
  .search input { border: 0; outline: 0; background: transparent; color: var(--ink); font: inherit; width: 100%; }
  .search svg { flex: none; opacity: 0.6; }

  .tabs { display: flex; gap: 0.4rem; margin-top: 1rem; flex-wrap: wrap; }
  .tab {
    border: 1px solid var(--border); background: var(--surface-1); color: var(--ink-2);
    border-radius: 999px; padding: 0.45rem 0.95rem; font-size: 0.85rem; font-weight: 600;
    cursor: pointer;
  }
  .tab.active { background: var(--ink); color: var(--page); border-color: var(--ink); }

  .legend-note { font-size: 0.78rem; color: var(--muted); margin-top: 0.6rem; display: flex; align-items: center; gap: 0.4rem; }
  .legend-note .swatch { width: 0.7rem; height: 0.7rem; border-radius: 3px; background: var(--good-bg); border: 1px solid var(--good); display: inline-block; }

  .table-scroll {
    margin-top: 1.2rem; overflow-x: auto; border: 1px solid var(--border);
    border-radius: 10px; background: var(--panel);
  }
  table { border-collapse: collapse; width: 100%; font-size: 0.86rem; }
  thead th {
    position: sticky; top: 0; background: var(--surface-1); text-align: left;
    padding: 0.65rem 0.9rem; font-weight: 700; white-space: nowrap;
    border-bottom: 1px solid var(--grid);
  }
  thead th.model-col { text-align: right; }
  thead th .th-label { display: inline-flex; align-items: center; gap: 0.4rem; justify-content: flex-end; width: 100%; }
  thead th .dot { width: 0.6rem; height: 0.6rem; border-radius: 50%; flex: none; }
  tbody td, tbody th {
    padding: 0.55rem 0.9rem; border-bottom: 1px solid var(--grid); white-space: nowrap;
  }
  tbody th { text-align: left; font-weight: 600; color: var(--ink-2); }
  tbody td {
    text-align: right; color: var(--ink);
    font-family: ui-monospace, "SF Mono", "Cascadia Code", Menlo, Consolas, monospace;
    font-variant-numeric: tabular-nums;
    font-size: 0.84rem;
  }
  tbody tr:last-child td, tbody tr:last-child th { border-bottom: none; }
  tbody tr:hover td, tbody tr:hover th { background: color-mix(in srgb, var(--ink) 4%, transparent); }
  td.best { background: var(--good-bg); color: var(--good); font-weight: 700; border-radius: 6px; }
  td.na { color: var(--muted); }

  .section-title { margin: 1.4rem 0 0.2rem; font-size: 0.95rem; font-weight: 700; }
  .section-sub { margin: 0 0 0.4rem; font-size: 0.78rem; color: var(--muted); }

  .empty { padding: 2rem; text-align: center; color: var(--muted); }

  @media (max-width: 640px) {
    .wrap { width: min(100%, calc(100% - 1.25rem)); padding-top: 1.25rem; }
    .search { flex: 1 1 100%; }
  }
</style>
</head>
<body>
  <div class="wrap">
    <header class="hero">
      <h1 class="brand">Holdout comparison browser</h1>
      <p class="lede" id="protocol-note"></p>
      <div class="toolbar">
        <div class="model-toggles" id="model-toggles"></div>
        <label class="search">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="7"/><path d="m21 21-4.3-4.3"/></svg>
          <input id="q" type="search" placeholder="Filter metrics…" autocomplete="off" />
        </label>
      </div>
      <div class="tabs" id="tabs"></div>
      <div class="legend-note"><span class="swatch"></span> highlighted = best value in that row (direction-aware; ties/neutral metrics unhighlighted)</div>
    </header>
    <main id="content"></main>
  </div>

  <script id="data" type="application/json">__EMBEDDED_JSON__</script>
  <script id="meta" type="application/json">__EMBEDDED_META__</script>
  <script>
    const DATA = JSON.parse(document.getElementById("data").textContent);
    const META = JSON.parse(document.getElementById("meta").textContent);
    const modelNames = META.order.filter((n) => DATA.baselines[n]);

    document.getElementById("protocol-note").textContent =
      `${DATA.protocol_note} · generated ${DATA.generated_at}`;

    // ---- model toggle chips ----
    const enabled = new Set(modelNames);
    const togglesEl = document.getElementById("model-toggles");
    for (const name of modelNames) {
      const chip = document.createElement("label");
      chip.className = "model-chip";
      chip.innerHTML = `
        <input type="checkbox" checked data-model="${name}" />
        <span class="dot" style="background:${META.colors[name]}"></span>
        <span>${META.labels[name] || name}</span>
      `;
      chip.querySelector("input").addEventListener("change", (e) => {
        if (e.target.checked) enabled.add(name); else enabled.delete(name);
        chip.classList.toggle("off", !e.target.checked);
        render();
      });
      togglesEl.appendChild(chip);
    }

    // ---- metric definitions: [label, path-getter, direction] ----
    // direction: "up" (higher better), "down" (lower better), null (neutral, no highlight)
    function get(obj, path) {
      return path.split(".").reduce((o, k) => (o == null ? undefined : o[k]), obj);
    }

    const PAIR_ROWS = [
      ["ROC-AUC", "pair.roc_auc", "up"],
      ["Average precision", "pair.average_precision", "up"],
      ["Best F1", "pair.best_f1", "up"],
      ["Best-F1 threshold", "pair.best_f1_threshold", null],
      ["Accuracy @ best-F1", "pair.best_f1_accuracy", "up"],
      ["Accuracy @ 0.5", "pair.accuracy_at_0.5", "up"],
      ["Mean cos (pos)", "pair.mean_cosine_positive", null],
      ["Mean cos (neg)", "pair.mean_cosine_negative", null],
      ["Mean cos gap", "pair.mean_cosine_gap", "up"],
      ["Std cos (pos)", "pair.std_cosine_positive", null],
      ["Std cos (neg)", "pair.std_cosine_negative", null],
      ["n pos", "pair.num_positive", null],
      ["n neg", "pair.num_negative", null],
    ];

    const RETRIEVAL_ROWS = [
      ["MRR", "retrieval.mrr", "up"],
      ["MAP", "retrieval.map", "up"],
      ["Mean rank", "retrieval.mean_rank", "down"],
      ["Median rank", "retrieval.median_rank", "down"],
      ["Top-1", "retrieval.top1", "up"],
      ["R@1", "retrieval.recall@1", "up"],
      ["R@5", "retrieval.recall@5", "up"],
      ["R@10", "retrieval.recall@10", "up"],
      ["NDCG@1", "retrieval.ndcg@1", "up"],
      ["NDCG@5", "retrieval.ndcg@5", "up"],
      ["NDCG@10", "retrieval.ndcg@10", "up"],
      ["P@1", "retrieval.precision@1", "up"],
      ["P@5", "retrieval.precision@5", "up"],
      ["P@10", "retrieval.precision@10", "up"],
      ["n queries", "retrieval.num_queries", null],
    ];

    const HARDNESS_ROWS = [
      ["Easy-neg AUC", "slices.neg_hardness.easy.pair_auc", "up"],
      ["Easy-neg n", "slices.neg_hardness.easy.n_negatives", null],
      ["Hard-neg AUC", "slices.neg_hardness.hard.pair_auc", "up"],
      ["Hard-neg n", "slices.neg_hardness.hard.n_negatives", null],
    ];

    const INTENTS = ["customers", "fundraise", "hiring", "partnerships", "other"];
    function intentRows() {
      const rows = [];
      for (const intent of INTENTS) {
        rows.push([`${intent} · n pairs`, `slices.intent.${intent}.n_pairs`, null]);
        rows.push([`${intent} · pair AUC`, `slices.intent.${intent}.pair_auc`, "up"]);
        rows.push([`${intent} · MRR`, `slices.intent.${intent}.mrr`, "up"]);
        rows.push([`${intent} · R@10`, `slices.intent.${intent}.recall@10`, "up"]);
      }
      return rows;
    }

    const TABS = [
      { id: "pair", label: "Pair", rows: PAIR_ROWS },
      { id: "retrieval", label: "Retrieval", rows: RETRIEVAL_ROWS },
      { id: "hardness", label: "Neg hardness", rows: HARDNESS_ROWS },
      { id: "intent", label: "Intent", rows: intentRows() },
    ];
    let activeTab = "pair";

    const tabsEl = document.getElementById("tabs");
    for (const t of TABS) {
      const btn = document.createElement("button");
      btn.className = "tab" + (t.id === activeTab ? " active" : "");
      btn.textContent = t.label;
      btn.addEventListener("click", () => {
        activeTab = t.id;
        document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        render();
      });
      tabsEl.appendChild(btn);
    }

    function fmt(v) {
      if (v === undefined || v === null) return null;
      if (typeof v === "boolean") return String(v);
      if (typeof v === "number") {
        if (Number.isInteger(v)) return String(v);
        return v.toFixed(4);
      }
      return String(v);
    }

    function render() {
      const q = document.getElementById("q").value.trim().toLowerCase();
      const activeModels = modelNames.filter((m) => enabled.has(m));
      const tab = TABS.find((t) => t.id === activeTab);
      const rows = tab.rows.filter((r) => r[0].toLowerCase().includes(q));

      const content = document.getElementById("content");
      if (!activeModels.length || !rows.length) {
        content.innerHTML = `<div class="table-scroll"><div class="empty">No ${!activeModels.length ? "models selected" : "metrics match that filter"}.</div></div>`;
        return;
      }

      let html = `<div class="table-scroll"><table><thead><tr><th>Metric</th>`;
      for (const m of activeModels) {
        html += `<th class="model-col"><span class="th-label"><span class="dot" style="background:${META.colors[m]}"></span>${META.labels[m] || m}</span></th>`;
      }
      html += `</tr></thead><tbody>`;

      for (const [label, path, direction] of rows) {
        const raw = activeModels.map((m) => get(DATA.baselines[m], path));
        const nums = raw.map((v) => (typeof v === "number" ? v : null));
        let bestIdx = -1;
        if (direction && nums.some((v) => v !== null)) {
          const valid = nums.map((v, i) => [v, i]).filter(([v]) => v !== null);
          const best = direction === "up"
            ? valid.reduce((a, b) => (b[0] > a[0] ? b : a))
            : valid.reduce((a, b) => (b[0] < a[0] ? b : a));
          const tie = valid.filter(([v]) => v === best[0]);
          if (tie.length === 1) bestIdx = best[1];
        }
        html += `<tr><th>${label}</th>`;
        raw.forEach((v, i) => {
          const display = fmt(v);
          if (display === null) {
            html += `<td class="na">—</td>`;
          } else {
            html += `<td class="${i === bestIdx ? "best" : ""}">${display}</td>`;
          }
        });
        html += `</tr>`;
      }
      html += `</tbody></table></div>`;
      content.innerHTML = html;
    }

    document.getElementById("q").addEventListener("input", render);
    render();
  </script>
</body>
</html>
"""


def build(in_path: Path, out_path: Path) -> None:
    data = json.loads(in_path.read_text())
    order = list(data["baselines"].keys())
    colors_light = {name: SERIES_COLORS_LIGHT[i % len(SERIES_COLORS_LIGHT)] for i, name in enumerate(order)}
    meta = {
        "order": order,
        "labels": {name: LABELS.get(name, name) for name in order},
        "colors": colors_light,
    }

    data_json = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    meta_json = json.dumps(meta, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")

    html = HTML_TEMPLATE.replace("__EMBEDDED_JSON__", data_json).replace("__EMBEDDED_META__", meta_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-path", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    build(args.in_path, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
